import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { StateAdapter } from '../../core/src/adapter.js';
import type { PersistedAgent } from '../../core/src/schemas.js';
import { agentCreatedMessage } from '../src/agentMessages.js';
import { restoredAgentInit, toPersistedAgent } from '../src/agentPersistence.js';
import { createAgentState } from '../src/agentState.js';
import { AgentStateStore } from '../src/agentStateStore.js';
import { claudeProvider } from '../src/providers/hook/claude/claude.js';
import {
  getProviderPreference,
  getRecommendedProvider,
  isAgentProvider,
  KEY_ASK_PROVIDER_EACH_TIME,
  KEY_DEFAULT_PROVIDER,
  KEY_PROVIDER_PREFERENCE_SET,
  PROVIDER_DEFINITIONS,
  saveProviderPreference,
} from '../src/providers/multiProvider.js';
import {
  processGeminiSessionSnapshot,
  processTranscriptLine,
  setHookProvider,
} from '../src/transcriptParser.js';

function memoryAdapter(initial: Record<string, unknown> = {}): StateAdapter {
  const settings = new Map<string, unknown>(Object.entries(initial));
  return {
    loadAgents: () => [],
    saveAgents: () => {},
    loadSeats: () => ({}),
    saveSeats: () => {},
    getSetting: <T>(key: string, defaultValue: T): T =>
      settings.has(key) ? (settings.get(key) as T) : defaultValue,
    setSetting: <T>(key: string, value: T) => {
      settings.set(key, value);
    },
  };
}

describe('multiProvider', () => {
  it('recognizes valid agent providers and recommends first installed', () => {
    expect(isAgentProvider('claude')).toBe(true);
    expect(isAgentProvider('codex')).toBe(true);
    expect(isAgentProvider('gemini')).toBe(true);
    expect(isAgentProvider('other')).toBe(false);

    expect(getRecommendedProvider({ claude: false, codex: true, gemini: true })).toBe('codex');
    expect(getRecommendedProvider({ claude: false, codex: false, gemini: true })).toBe('gemini');
    expect(getRecommendedProvider({ claude: false, codex: false, gemini: false })).toBeNull();
  });

  it('defaults askEachTime to true on first run when multiple providers are installed', () => {
    const adapter = memoryAdapter();
    const pref = getProviderPreference(adapter, { claude: true, codex: true, gemini: false });
    expect(pref.defaultProvider).toBe('claude');
    expect(pref.askEachTime).toBe(true);
    expect(pref.preferenceSet).toBe(false);
  });

  it('persists provider preference and respects saved askEachTime=false', () => {
    const adapter = memoryAdapter();
    saveProviderPreference(adapter, 'codex', false);
    expect(adapter.getSetting(KEY_DEFAULT_PROVIDER, null)).toBe('codex');
    expect(adapter.getSetting(KEY_ASK_PROVIDER_EACH_TIME, true)).toBe(false);
    expect(adapter.getSetting(KEY_PROVIDER_PREFERENCE_SET, false)).toBe(true);

    const pref = getProviderPreference(adapter, { claude: true, codex: true, gemini: true });
    expect(pref.defaultProvider).toBe('codex');
    expect(pref.askEachTime).toBe(false);
    expect(pref.preferenceSet).toBe(true);
  });

  it('defines launch commands and session formats for claude, codex, gemini', () => {
    expect(PROVIDER_DEFINITIONS.claude.sessionFormat).toBe('jsonl');
    expect(PROVIDER_DEFINITIONS.codex.sessionFormat).toBe('jsonl');
    expect(PROVIDER_DEFINITIONS.gemini.sessionFormat).toBe('gemini-json');

    expect(PROVIDER_DEFINITIONS.codex.buildLaunchCommand('s1', '/tmp').command).toBe('codex');
    expect(PROVIDER_DEFINITIONS.gemini.buildLaunchCommand('s1', '/tmp').command).toBe('gemini');
  });

  it('persists and restores provider metadata with claude fallback for legacy agents', () => {
    const codexAgent = createAgentState({
      id: 1,
      sessionId: 's-codex',
      projectDir: '/tmp',
      jsonlFile: '/tmp/rollout-1.jsonl',
      isExternal: false,
      providerId: 'codex',
      sessionFormat: 'jsonl',
    });
    const persisted = toPersistedAgent(codexAgent);
    expect(persisted?.provider).toBe('codex');
    expect(persisted?.sessionFormat).toBe('jsonl');

    const msg = agentCreatedMessage(1, codexAgent);
    expect(msg.provider).toBe('codex');

    const legacy: PersistedAgent = {
      id: 2,
      terminalName: 'Claude Code #1',
      jsonlFile: '/tmp/legacy.jsonl',
      projectDir: '/tmp',
    };
    const restored = restoredAgentInit(legacy);
    expect(restored.providerId).toBe('claude');
    expect(restored.sessionFormat).toBe('jsonl');
  });
});

describe('multiProvider transcript parsing', () => {
  let store: AgentStateStore;
  let broadcasts: Array<Record<string, unknown>>;
  let waitingTimers: Map<number, ReturnType<typeof setTimeout>>;
  let permissionTimers: Map<number, ReturnType<typeof setTimeout>>;

  beforeEach(() => {
    vi.useFakeTimers();
    setHookProvider(claudeProvider);
    store = new AgentStateStore();
    broadcasts = [];
    store.on('broadcast', (m: Record<string, unknown>) => broadcasts.push(m));
    waitingTimers = new Map();
    permissionTimers = new Map();
  });

  afterEach(() => {
    for (const t of waitingTimers.values()) clearTimeout(t);
    for (const t of permissionTimers.values()) clearTimeout(t);
    vi.useRealTimers();
  });

  it('parses Codex function_call, function_call_output, and task_complete', () => {
    const agent = createAgentState({
      id: 1,
      sessionId: 'codex-1',
      projectDir: '/tmp',
      jsonlFile: '/tmp/rollout-1.jsonl',
      isExternal: false,
      providerId: 'codex',
      sessionFormat: 'jsonl',
    });
    store.set(1, agent);

    processTranscriptLine(
      1,
      JSON.stringify({
        type: 'response_item',
        payload: {
          type: 'function_call',
          call_id: 'call_1',
          name: 'shell_command',
          arguments: JSON.stringify({ command: 'ls -la' }),
        },
      }),
      store,
      waitingTimers,
      permissionTimers,
    );
    expect(agent.activeToolIds.has('call_1')).toBe(true);
    expect(broadcasts).toContainEqual({
      type: 'agentToolStart',
      id: 1,
      toolId: 'call_1',
      status: 'Running: ls -la',
    });

    processTranscriptLine(
      1,
      JSON.stringify({
        type: 'response_item',
        payload: { type: 'function_call_output', call_id: 'call_1', output: 'ok' },
      }),
      store,
      waitingTimers,
      permissionTimers,
    );
    expect(agent.activeToolIds.has('call_1')).toBe(false);
    vi.advanceTimersByTime(350);
    expect(broadcasts).toContainEqual({ type: 'agentToolDone', id: 1, toolId: 'call_1' });

    processTranscriptLine(
      1,
      JSON.stringify({ type: 'event_msg', payload: { type: 'task_complete' } }),
      store,
      waitingTimers,
      permissionTimers,
    );
    expect(agent.isWaiting).toBe(true);
    expect(broadcasts).toContainEqual({ type: 'agentStatus', id: 1, status: 'waiting' });
  });

  it('parses Gemini session snapshots for user turns and toolCalls', () => {
    const agent = createAgentState({
      id: 2,
      sessionId: 'gem-1',
      projectDir: '/tmp',
      jsonlFile: '/tmp/session-1.json',
      isExternal: false,
      providerId: 'gemini',
      sessionFormat: 'gemini-json',
    });
    store.set(2, agent);

    processGeminiSessionSnapshot(
      2,
      JSON.stringify({
        messages: [
          {
            id: 'm1',
            type: 'user',
            timestamp: '2026-01-01T00:00:00Z',
            toolCalls: [
              {
                id: 'tc1',
                name: 'read_file',
                args: { file_path: '/src/app.ts' },
                status: 'running',
              },
            ],
          },
        ],
      }),
      store,
      waitingTimers,
    );
    expect(agent.processedGeminiMessages).toBe(1);
    expect(agent.activeToolIds.has('gemini-turn:m1')).toBe(true);
    expect(agent.activeToolIds.has('tc1')).toBe(true);
    expect(broadcasts).toContainEqual({
      type: 'agentToolStart',
      id: 2,
      toolId: 'tc1',
      status: 'Reading app.ts',
    });

    processGeminiSessionSnapshot(
      2,
      JSON.stringify({
        messages: [
          {
            id: 'm1',
            type: 'user',
            timestamp: '2026-01-01T00:00:00Z',
            toolCalls: [
              {
                id: 'tc1',
                name: 'read_file',
                args: { file_path: '/src/app.ts' },
                status: 'running',
              },
            ],
          },
          {
            id: 'm2',
            type: 'gemini',
            timestamp: '2026-01-01T00:00:05Z',
            toolCalls: [
              {
                id: 'tc1',
                name: 'read_file',
                args: { file_path: '/src/app.ts' },
                status: 'success',
              },
            ],
          },
        ],
      }),
      store,
      waitingTimers,
    );
    expect(agent.processedGeminiMessages).toBe(2);
    expect(agent.activeToolIds.has('gemini-turn:m1')).toBe(false);
    expect(agent.activeToolIds.has('tc1')).toBe(false);
    vi.advanceTimersByTime(350);
    expect(broadcasts).toContainEqual({ type: 'agentToolDone', id: 2, toolId: 'gemini-turn:m1' });
    expect(broadcasts).toContainEqual({ type: 'agentToolDone', id: 2, toolId: 'tc1' });
  });
});
