import test from 'node:test';
import assert from 'node:assert/strict';
import {statusGroup, queueMessageKey} from '../../packages/videotranslator/videotranslator/web/history.js';
import {historyCopy} from '../../packages/videotranslator/videotranslator/web/history-copy.js';

test('Every backend state has an explicit user-facing history category', () => {
  const expected = {uploaded:'waiting',inspecting:'running',awaiting_credits:'waiting',awaiting_capacity:'queued',queued:'queued',submitting:'running',provisioning:'running',running:'running',cancelling:'running',succeeded:'succeeded',failed:'failed',cancelled:'cancelled',expired:'expired'};
  for (const [status,group] of Object.entries(expected)) assert.equal(statusGroup(status),group,status);
});

test('Paused service is not described as a full queue', () => {
  assert.equal(queueMessageKey({capacity_wait_reason:'gpu_starts_disabled'}),'translationPaused');
  assert.equal(queueMessageKey({capacity_wait_reason:'cost_budget_exceeded'}),'translationPaused');
  assert.equal(queueMessageKey({capacity_wait_reason:'workers_busy'}),'awaiting_capacity');
  assert.equal(queueMessageKey({}),'awaiting_capacity');
});

test('History labels exist in each supported interface language', () => {
  for (const locale of ['en','zh','fr','es','de','ja','ko','pt']) {
    assert.deepEqual(Object.keys(historyCopy[locale]),Object.keys(historyCopy.en));
    assert.ok(Object.values(historyCopy[locale]).every(value => typeof value === 'string' && value.length));
  }
});
