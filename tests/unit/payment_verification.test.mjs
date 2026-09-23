import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyReturnedPayment} from '../../packages/videotranslator/videotranslator/web/payment-verification.js';
import {createAccount} from '../../packages/videotranslator/videotranslator/web/account.js';

test('Stripe return reconciles the existing payment without starting a checkout', async () => {
  const calls = [];
  const result = await verifyReturnedPayment({paymentId:'pay_1', api:async (path,body) => {
    calls.push([path,body]); return {provider:'stripe',status:body ? 'succeeded' : 'pending'};
  }});
  assert.equal(result.status,'succeeded');
  assert.deepEqual(calls,[['/billing/payments/pay_1',undefined],['/billing/payments/pay_1/reconcile',{}]]);
});

test('Pending status and stalled network both leave the spinner within a bound', async () => {
  await assert.rejects(verifyReturnedPayment({paymentId:'pay_1',attempts:2,sleep:async()=>{},
    api:async()=>({provider:'stripe',status:'pending'})}), /paymentPendingLong/);
  await assert.rejects(verifyReturnedPayment({paymentId:'pay_1',timeoutMs:5,api:()=>new Promise(()=>{})}), /paymentCheckUnavailable/);
});

test('An account change abandons verification; PayPal requires the matching order', async () => {
  let calls = 0;
  assert.equal(await verifyReturnedPayment({paymentId:'pay_1',isCurrent:()=>false,
    api:async()=>{calls++;return {provider:'stripe',status:'pending'};}}),null);
  assert.equal(calls,1);
  await assert.rejects(verifyReturnedPayment({paymentId:'pay_1',paypalToken:'wrong',
    api:async()=>({provider:'paypal',status:'pending',provider_order_id:'actual'})}), /paymentMismatch/);
});

test('Return dialog hides payment form while checking and after success', async t => {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id,{hidden:false,disabled:false,textContent:'',dataset:{},
      setAttribute(){},showModal(){this.open=true;},close(){this.open=false;}});
    return elements.get(id);
  };
  const storage = new Map();
  const original = {};
  const globals = {
    document:{getElementById:element,documentElement:{lang:'en'},querySelectorAll:()=>[],addEventListener(){},dispatchEvent(){}},
    window:{addEventListener(){}}, location:{search:'?checkout=return&payment_id=pay_1',pathname:'/'},
    history:{replaceState(){globals.location.search='';}},
    localStorage:{getItem:()=> 'u1',setItem(){}},
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
  };
  for (const [key,value] of Object.entries(globals)) {original[key]=globalThis[key];globalThis[key]=value;}
  t.after(()=>{for(const key of Object.keys(globals)) {if(original[key]===undefined) delete globalThis[key];else globalThis[key]=original[key];}});
  t.mock.method(globalThis,'setInterval',()=>0);
  let resolvePayment;
  const pending = new Promise(resolve=>{resolvePayment=resolve;});
  const account = await createAccount({config:{auth_mode:'demo',profile:'test',payment_providers:['stripe']},
    api:async path=>path==='/me'?{balance_cents:166}:pending,t:k=>k,changed:async()=>{},report:()=>{}});
  const started=account.start();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(element('topup-form').hidden,true);
  assert.equal(element('payment-status').textContent,'paymentPending');
  assert.equal(element('payment-done').hidden,true);
  resolvePayment({provider:'stripe',status:'succeeded'});
  await started;
  assert.equal(element('topup-form').hidden,true);
  assert.equal(element('topup-title').textContent,'paymentSuccessTitle');
  assert.equal(element('payment-done').hidden,false);
  assert.equal(element('payment-retry').hidden,true);
});
