import test from 'node:test';
import assert from 'node:assert/strict';
import {createErrorNotice} from '../../packages/videotranslator/videotranslator/web/error-notice.js';

const setup = () => {
  const element = {hidden:true,textContent:''};
  return {element, notice:createErrorNotice({element,t:key=>key,isRunning:()=>true})};
};

test('Successful authenticated requests clear stale re-login notices', () => {
  const {element,notice} = setup();
  notice.show(new Error('unauthenticated'),{background:true,source:'poll'});
  assert.equal(element.textContent,'jobStatusReconnecting');
  notice.authenticated();
  assert.equal(element.hidden,true); assert.equal(element.textContent,'');
});

test('An actual persistent credential rejection still asks for sign-in without implying job failure', () => {
  const {element,notice} = setup();
  for (let n=0;n<3;n++) notice.show(new Error('unauthenticated'),{background:true,source:'poll'});
  assert.equal(element.textContent,'jobAuthRequired');
  notice.authenticated();
  notice.show(new Error('unauthenticated'),{background:true,source:'poll'});
  assert.equal(element.textContent,'jobStatusReconnecting');
});

test('Recovered job polling clears its network warning, not an action error', () => {
  const {element,notice} = setup();
  notice.show(new Error('network'),{background:true,source:'poll'});
  notice.authenticated(); assert.equal(element.hidden,false);
  notice.recovered('balance'); assert.equal(element.hidden,false);
  notice.recovered('poll'); assert.equal(element.hidden,true);
  notice.show(new Error('insufficient_credits'));
  notice.authenticated(); notice.recovered('poll');
  assert.equal(element.hidden,false); assert.equal(element.textContent,'insufficient_credits');
});

test('A provider outage never turns into a request to log in', () => {
  const {element,notice} = setup();
  for(let n=0;n<5;n++) notice.show(new Error('auth_unavailable'),{background:true,source:'poll'});
  assert.equal(element.textContent,'jobStatusReconnecting');
});
