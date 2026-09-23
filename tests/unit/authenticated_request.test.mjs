import test from 'node:test';
import assert from 'node:assert/strict';
import {authenticatedFetch} from '../../packages/videotranslator/videotranslator/web/authenticated-request.js';

const reply = (status, code) => new Response(JSON.stringify({detail:{code}}), {status});
const base = {getToken:async () => 'test', isCurrent:() => true, sleep:async () => {}};

test('Expired credential is refreshed once before resending the protected request', async () => {
  const tokens = [], requests = [];
  const result = await authenticatedFetch('/confirm', {method:'POST',body:'unchanged'}, {
    ...base, getToken:async force => {tokens.push(force); return force ? 'new' : 'old';},
    fetchImpl:async (_, options) => {
      requests.push(options);
      return requests.length === 1 ? reply(401,'unauthenticated') : reply(200);
    },
  });
  assert.equal(result.status,200);
  assert.deepEqual(tokens,[false,true]);
  assert.equal(requests[1].headers.Authorization,'Bearer new');
  assert.equal(requests[1].body,'unchanged');
});

test('Temporary verification failures recover with a bounded retry', async () => {
  let calls = 0;
  const result = await authenticatedFetch('/me', {}, {...base,
    fetchImpl:async () => ++calls < 3 ? reply(503,'auth_unavailable') : reply(200)});
  assert.equal(result.status,200); assert.equal(calls,3);
});

test('Persistent auth failure cannot loop', async () => {
  for (const [status,code,max] of [[401,'unauthenticated',2],[503,'auth_unavailable',3]]) {
    let calls = 0;
    const response = await authenticatedFetch('/me', {}, {...base,
      fetchImpl:async () => {calls++; return reply(status,code);}});
    assert.equal(response.status,status); assert.equal(calls,max);
  }
});

test('General service errors and transport failures never replay a mutation', async () => {
  let calls = 0;
  const result = await authenticatedFetch('/confirm', {method:'POST'}, {...base,
    fetchImpl:async () => {calls++; return reply(503,'backend_failed');}});
  assert.equal(result.status,503); assert.equal(calls,1);
  calls = 0;
  await assert.rejects(authenticatedFetch('/confirm', {method:'POST'}, {...base,
    fetchImpl:async () => {calls++; throw new Error('connection lost');}}));
  assert.equal(calls,1);
});

test('Account switch during retry cannot send the previous account request', async () => {
  let current = true, calls = 0;
  await assert.rejects(authenticatedFetch('/confirm', {method:'POST'}, {...base,
    isCurrent:() => current, sleep:async () => {current = false;},
    fetchImpl:async () => {calls++; return reply(503,'auth_unavailable');}}), /unauthenticated/);
  assert.equal(calls,1);
});
