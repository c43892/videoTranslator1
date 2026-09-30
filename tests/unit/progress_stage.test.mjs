import assert from 'node:assert/strict';
import test from 'node:test';
import {progressStageText} from '../../packages/videotranslator/videotranslator/web/history.js';

const copy = {
  currentStage: 'Current stage', stageSynthesize: 'Generating dubbed speech',
  stageProvisioning: 'Preparing processing resources', stageProcessing: 'Processing media',
};
const t = key => copy[key] || key;

test('Progress text names the current stage and percentage', () => {
  assert.equal(
    progressStageText({status:'running', stage:'synthesize', progress_percent:77}, t),
    'Current stage: Generating dubbed speech · 77%',
  );
  assert.equal(
    progressStageText({status:'provisioning', stage:'', progress_percent:4}, t),
    'Current stage: Preparing processing resources · 4%',
  );
  assert.equal(
    progressStageText({status:'running', stage:'future_stage', progress_percent:140}, t),
    'Current stage: Processing media · 100%',
  );
});
