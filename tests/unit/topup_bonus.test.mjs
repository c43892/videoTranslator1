import test from 'node:test';
import assert from 'node:assert/strict';
import {packageLabel} from '../../packages/videotranslator/videotranslator/web/topup-packages.js';
import {accountCopy} from '../../packages/videotranslator/videotranslator/web/account-copy.js';

test('Top-up labels distinguish payment from credited balance in both languages', () => {
  const money = cents => `$${cents/100}`;
  for (const locale of ['en','zh']) {
    const t = key => accountCopy[locale][key];
    for (const [paid,credit,bonus] of [[100,100,0],[1000,1100,10],[5000,6000,20],[10000,13000,30]]) {
      const label = packageLabel({amount_minor:paid,point_units:credit},money,t);
      assert.ok(label.includes(money(paid)) && label.includes(money(credit)));
      assert.equal(label.includes('%'),bonus>0);
      if(bonus) assert.ok(label.includes(`${bonus}%`));
      assert.ok(!label.includes('undefined') && !label.includes('{'));
    }
  }
});
