# Local Studio and GPU engine

The Studio at `http://localhost:8090` now runs in the
`videotranslator-studio-studio-1` container. Its `DockerEngineBackend` submits to
the existing `videotranslator-api-1` engine container. The engine's PostgreSQL
queue, Celery worker (`--concurrency=1`) and deployment-wide PostgreSQL advisory
lock serialize video and audio processing, including work submitted by the old
8080 interface.

The real path uses Demucs and OpenAI Whisper without speaker identification,
DeepSeek translation/punctuation, local GPU IndexTTS2, timeline alignment and
background mixing. A failed segment retains the original dialogue with a
visible warning. Complete job failures return the actual debit once to account
balance. Stripe/PayPal cash refunds are never invoked.

Successful synthesized dialogue always fits its original start/end window using
pitch-preserving FFmpeg `atempo`, chaining stages no greater than 2x. The former
1.25x threshold no longer triggers original-language fallback. Short speech is
padded; translations are not shortened or resynthesized just for duration.
Extreme compression can reduce intelligibility, but does not reject valid speech.
The `max_speedup` engine argument remains only for compatibility with stored config.
The standalone duration matcher uses the same forced-fit policy.

`deploy/engine-patches/repair_duration_overflow.py` remixes completed engine tasks
from cached synthesis into separate `corrections/tempo-v1` artifacts. Validate and
publish with `deploy/publish_duration_correction.py` inside Studio, then activate
the engine output using the repair script's `--activate` switch. Previous media
and the original manifest remain available; no model calls or extra charges occur.

## Start Studio

The existing GPU engine must be running and configured first. Then:

```powershell
docker compose -p videotranslator-studio -f compose.chat.yml -f compose.engine.yml up -d --build studio
```

`compose.engine.yml` enables `APP_PROFILE=local-full`, `ENGINE_BACKEND=docker`,
and checks engine readiness at startup. Standalone `compose.chat.yml` remains a
non-processing preview. Native launches support `DOCKER_COMMAND` for a CLI that
is not on PATH. On this machine Docker Desktop uses a per-user installation.

The override binds `vt-data/chat-preview` to `/data`, preserving the existing
Firebase users, sandbox balance ledger, conversations, job history and media.
The active DB is `store-stripe-sandbox.db`; `store-before-container.db` is the
pre-migration backup. Do not run the old native server on 8090 simultaneously.
The native Stripe listener still forwards to 8090; its endpoint did not change.

This trusted local controller mounts the Docker socket. It must remain bound
to localhost and must not be used as an untrusted hosted workload. Cloud
deployments should replace the adapter with their managed job backend rather
than exposing a Docker socket.

## Engine source and persistent patch

The running engine comes from the existing `codex/fresh-start` branch, commit
`3f196f8`, rather than the unrelated older code under `src/volumn`. Its missing
worktree was restored at `vt-data/engine-source`:

```powershell
git worktree add --detach vt-data/engine-source codex/fresh-start
```

Skip that command if the worktree already exists. Its ignored `.env` was restored
from the existing containers' settings without logging credentials. The patch
in `deploy/engine-patches/whisper-empty-words.patch` filters empty Whisper
alignment entries without losing real words or their timestamps. Its regression
test is run in the patched image build:

```powershell
docker build -f deploy/Dockerfile.engine-patched -t videotranslator-app:studio .
docker compose -p videotranslator -f vt-data/engine-source/compose.yaml -f deploy/compose.engine-worker.yml up -d --no-deps --no-build worker
```

Recreate the worker only while no task is active. This updates the worker image
without changing the existing GPU/model service, queues or volumes.

## Contracts and result validation

- Each immutable Studio job specification maps to a deterministic engine UUID.
  Submission retries recover the same engine job after timeouts/restarts. The
  internal non-interactive engine principal has no issued browser login session.
- Firebase ownership checks stay at the Studio boundary. Browser requests never
  receive Docker access, engine credentials or provider keys.
- Audio uploads are wrapped with a duration-matched private video track solely
  for the engine's timeline contract; delivered output is a 192 kbps MP3.
- Video outputs are exported as H.264/yuv420p + AAC with faststart. Stream and
  duration checks plus full FFmpeg decoding run before atomic publication and
  success. No placeholder or partially copied file is published.
- Authenticated result requests mint short-lived signed playback URLs. Native
  HTTP Range playback supports seeking; downloads reuse that authorization.
  `LOCAL_STORAGE_SIGNING_SECRET` is private and persistent in `.env`.
- Segment warnings are retained in task records and shown in conversation and
  history. A partial translation must not be presented as warning-free.

## Verified on 2026-09-22

The user's `dtlNRia89_Q` YouTube source was translated into Chinese as task
`job_d88c7417f2894c16`: 60.394-second H.264/AAC MP4, 30,765,379 bytes,
all 10 dialogue segments replaced, no segment warnings. The prior failed ASR
attempt returned $0.11 once; the successful attempt consumed $0.11 of sandbox
balance. A separate 14-second audio integration test also produced a valid MP3
without segment warnings. It waited while the video occupied the GPU queue.
External Chrome played the video with `readyState=4`, no media error, and a
progressing playback position. Range authorization, submission retries,
fast completion, warnings and invalid-media handling have regression coverage.

Payment mode remains **sandbox**. Real translation is enabled independently of
live payment activation.

## Current UI entry point (2026-09-23)

The patched engine API uses `STUDIO_PUBLIC_URL` from
`deploy/compose.engine-worker.yml` to redirect `/` and `/index.html` on port 8080
to the Firebase-backed Studio on port 8090. The old Chinese-only email/password
screen is no longer the bookmarked entry point. Query parameters are not
forwarded. Engine API paths, authentication and worker communication are unchanged.
Apply the overlay to `api` as well as `worker` after building the patched image.

## Dialogue residue correction (2026-09-22)

The engine overlay now includes `dialogue-residue.patch` and waveform regressions.
The dubbed dialogue timeline starts from silence and includes only synthesized
speech plus explicitly flagged original fallback intervals. Previously, the entire
original dialogue stem was retained outside ASR segment windows, leaking source
speech at gaps and boundaries. Overlapping fallback windows are copied once;
overlapping translated speakers still accumulate. Music/effects are unchanged.
Demucs leakage within the background stem remains a model limitation; this change
does not claim perfect source separation or suppress the entire background.

Both completed Studio videos (`job_cf0372a0fb30cc7c`, `job_d88c7417f2894c16`)
were remixed from their existing stems and synthesized speech, without provider
calls or additional balance deductions. Published keys are `result-dialogue-v2.mp4`;
the original outputs remain available on disk. Their evidence sidecars record
the source manifest, prior output key, and zero new dialogue signal in uncovered
intervals (3.31 seconds and 10.61 seconds respectively). Full FFmpeg decoding and
duration/stream validation passed before publishing.

For a future repair, run `deploy/engine-patches/remix_completed_dialogue.py`
inside the patched engine with its engine UUID, then run
`deploy/publish_dialogue_correction.py` inside Studio with the Studio job ID.
These tools preserve previous artifacts and never create tasks or ledger entries.

## Switchable translated subtitles (2026-09-23)

`soft-subtitles.patch` writes translated SRT/VTT before final assembly and embeds
the SRT as a separate MP4 `mov_text` stream, tagged `zho` or `eng`. Cue times use
the original dialogue start/end timestamps, matching the aligned dubbed audio.
Captions are enabled by default and can be disabled in a compatible player.
Videos without dialogue still export without a subtitle stream.

Studio preserves subtitle streams when producing its browser-compatible MP4.
It also copies translated WebVTT to `<output_object_key>.vtt` for the HTML video
player, since browser support for embedded MP4 captions varies. The authenticated
result endpoint returns `subtitle_url` and `subtitle_language` when available;
the signed playback route binds each URL to either the media or subtitle asset.
Ownership, expiration, deletion, and successful-task restrictions apply to both.
Deleting a result also deletes its WebVTT sidecar. Existing results without a
subtitle asset remain playable.

For a completed video, run `deploy/publish_subtitles.py <Studio job ID>` inside
Studio. It uses that task's existing translated subtitles and current published
video (including prior audio corrections), copies audio/video without encoding,
verifies their packet hashes, and publishes a new `*-subtitles-v1.mp4` result.
Previous artifacts are preserved; no provider calls or ledger entries are made.
This was applied to `job_5912d88b4dd139e8` and `job_ae54d7ef67d527dd`.


## Failure investigation: job_aa0b3b34c29329ff (2026-09-23)

Engine `80b9f2dc-69f1-5c08-9c31-72891fa77d9c` failed in the mix
stage (83%) with `ValueError: Invalid original fallback interval`. Studio's last
polled progress still showed translate/45%. `seg-00005` had start=end=644.0599976,
was flagged `zero_duration_utterance`, and was routed to original-speech fallback.
`media.dialogue_timeline` rejects its empty interval and aborts the full result.

There is also an upstream quality issue: Whisper returned only five punctuation
marks for a 660.901-second clip. Its word/text punctuation mapping succeeded,
but the optional DeepSeek punctuation batch was rejected for source inconsistency.
The fallback therefore retained sparse punctuation; segmentation produced just six
utterances, including spans 0–130.28, 130.46–276.86, and 276.86–604.94 seconds.
These cross speakers and exceed reference/text limits. All six segments fell back
to original speech; none synthesized translated audio. Merely suppressing the
zero-interval exception would not produce an acceptable dubbed result.

The original artifacts and manifest are preserved. This investigation did not
rerun providers or alter the failed job. Its 111-cent charge was automatically
credited back to the account. Follow-up pipeline work needs to address sparse
punctuation recovery, speaker-boundary handling, and zero-duration fallback.

### Segmentation recovery implementation

`segmentation-recovery.patch` introduces `punctuation-min3-v2`. Separation and
Whisper caches are retained, but new segment IDs, translations, references and
synthesis outputs live under the new version so they cannot reuse incompatible
old segment artifacts. The punctuator requests token-indexed punctuation marks,
preserving source characters and word timestamps locally. Requests are bounded
to 160 tokens / 1800 source characters and malformed batches retry in smaller
pieces once; only an exhausted piece falls back to its original punctuation.

A confirmed speaker change now ends a clause even without punctuation. The
same-speaker short-clause merging rule remains; unknown labels inside words do
not create artificial speaker changes. Point-timed words can merge naturally
with same-speaker neighbors. An isolated zero-length fallback remains flagged in
the manifest, contributes no audio samples and emits no invalid subtitle cue.
Other translated segments can still complete. A job with segments but no actual
translated audio now raises a clear review error instead of publishing original
speech as a successful dubbed result.

`validate_cached_segmentation.py <engine job ID>` checks the failed task's cached
ASR using the new punctuator and writes a versioned validation report, without
resubmitting a job or changing the ledger. The container image runs 54 regressions
covering segmentation, punctuation, real media assembly, fallbacks, time fitting,
subtitle embedding and resumable pipeline behavior.

Validation of the failed task's 1,402 cached words produced 86 segments (formerly
6), preserving every source character and original word timestamp. No zero-length
utterance or mixed-speaker flag remained. All five known speakers have duration-
eligible reference candidates. Three unknown-speaker segments and four overlong
utterances remain explicitly flagged; the longest is 22.32 seconds. One punctuation
sub-batch used its source punctuation after the bounded retry. This validates
segmentation, not the final synthesized audio; the failed Studio job was not
restarted or charged by this validation.


## Current policy: each utterance supplies its own reference (2026-09-23)

This policy supersedes the identity/reference eligibility rules described in the
historical investigations above. The active worker uses `WhisperTranscriber`;
there is no diarization request, identity decision, or stable speaker bank.
`punctuation-min3-v3-local-reference` splits on punctuation and merges short
neighbors without consulting legacy speaker labels. The empty `speaker_id`
field remains only for stored-schema compatibility. Old user overrides are
version-isolated; speaker reference overrides are no longer used.

Both `spk_audio_prompt` and `emo_audio_prompt` use the current utterance's original
audio. Long utterances retain their complete text/timeline; only their conditioning
reference uses a contiguous energy-selected excerpt from that same audio (up to
15 seconds). Short excerpts are silence-padded to one second for feature extraction,
without borrowing other speech. Reference quality/identity no longer veto synthesis.
Unrecognized text and an actual inability to generate usable audio are fallback
conditions. Entire recordings with no recognized text retain their dialogue stem.

The 120-token setting controls IndexTTS2's internal chunks, not an utterance-level
rejection threshold. Full translation text is passed to the model. Generation-cap
warnings retry with smaller internal chunks before reporting synthesis failure.
Successful speech is always pitch-preserving time-fitted to its original window.
The private GPU service still serializes requests, and the worker retains its
shared execution lock. Long text does not create concurrent GPU work.

Punctuation responses can contain valid suggestions mixed with Japanese letters
such as “が” or “の” erroneously emitted as marks. Only the invalid suggestions
are discarded; valid punctuation and all source text/timestamps are preserved.
Conflicting suggestions for one position are discarded instead of guessed.

Build `deploy/Dockerfile.engine-patched` as `videotranslator-app:studio` and
`deploy/Dockerfile.tts-patched` as `videotranslator-indextts2:studio`, then apply
`deploy/compose.engine-worker.yml` to api, worker and tts with no jobs running.
The final engine layer runs 53 regression tests, including source reference
provenance, long-text acceptance, no diarization request, punctuation recovery,
real FFmpeg timing/mixing and switchable subtitles.

`repair_segment_references.py` resynthesizes a completed task's cached translations
under the shared GPU execution lock and saves new artifacts separately. It keeps
all original artifacts, uses each original segment for both prompts and validates
output audio alignment and complete video decoding. `publish_segment_correction.py`
checks identity/duration/codecs/subtitles, atomically selects the new result on the
same Studio task, and never creates ledger entries or new charges.

Validated and republished Studio `job_b55f537c25a75bc9` / engine
`f2be0288-d056-52ae-868d-6c02c6176ebf`: all 85 cached utterances were resynthesized,
including all 11 former fallbacks. Both prompt waveforms were checked against
that utterance's own source samples; only same-segment cropping or zero padding
is permitted. Source text, translations and original timestamps stayed identical.
Every aligned WAV has exactly round(window_seconds * 48000) frames. The MP4
passes full decoding and retains embedded mov_text plus a WebVTT sidecar.
There are zero original-audio fallbacks and zero result warnings. Historical
punctuation diagnostics remain in the correction manifest and previous artifacts.
No new task or balance debit was created.

This validation exposed a sample-rate rounding edge case (Whisper's
5.10003662109375-second window). Alignment now resamples to 48 kHz before padding
and trimming by integer sample count, avoiding excess end samples without
rejecting successful synthesis. A regression covers that exact window.


## Scribe and preservation of non-dialogue sounds (2026-09-23)

`scribe-events.patch` adds an independent ElevenLabs Scribe adapter. Configure the
root `.env` with `TRANSCRIPTION_PROVIDER=scribe`, `SCRIBE_MODEL=scribe_v2` and
`ELEVENLABS_API_KEY` (never commit its value). Whisper remains available by selecting
`TRANSCRIPTION_PROVIDER=whisper`. Apply the engine Compose stack with the explicit
root environment file:

```powershell
docker compose --env-file .env -p videotranslator -f vt-data/engine-source/compose.yaml -f deploy/compose.engine-worker.yml up -d --no-deps --no-build api worker
```

Scribe receives 16 kHz mono FLAC with word timestamps, audio-event tagging and
verbatim recognition enabled, and diarization disabled. Events are stored
separately from words and never sent as dialogue to translation or synthesis.
Provider/model configuration separates recognition and pipeline caches.
`punctuation-min3-v4-audio-events` splits at intervening sound events and prevents
short-clause merging across them. The reference policy remains segment-local.

The current mixer supersedes the historical silence-gap policy: it starts from
the original dialogue stem and replaces only successfully synthesized intervals.
Uncovered sounds and failed utterances keep their original samples. Explicit
audio-event intervals are restored even when an event overlaps recognized speech;
this avoids deleting the event, but overlapping original speech can remain too.
Timestamp/event detection is not perfect acoustic separation.

The final image layer runs 64 regression tests covering both recognizers, cache
selection, event boundaries, actual waveform preservation, original fallback,
overlapping events, timing, pipeline propagation and switchable subtitles.
`deploy/engine-patches/validate_scribe.py JOB_ID`, executed inside the configured
engine container, recognizes an existing dialogue stem and writes a separate
validation transcript/segments/summary without modifying the completed result or
creating a Studio balance debit. The external transcription call uses API credit.

Live validation with `scribe_v2` succeeded on the completed 660.901-second Japanese
sample: 2,262 word entries, 88 initial segments, 19.13 seconds elapsed, diarization
disabled. No audio events were returned for this recording; therefore this sample
does not establish event-detection recall. The event/mixing behavior is covered by
controlled waveform tests. The validation did not rerun translation/TTS or replace
the existing downloadable result. New queued jobs use Scribe after the API/worker
restart. Secrets remain only in ignored local configuration.


## Voice-change boundaries without role identities (2026-09-23)

This supersedes the preceding `diarize=false` setting. Scribe is now called with
`diarize=true` solely to extract times where adjacent voice labels change. The
normalized transcript retains only `turn_boundaries`, empty compatibility speaker
fields, and `turn_detection=boundary_only`; no role registry, recurring voice
reference bank or unknown-identity rejection is used. The raw provider response
is retained for diagnostics. Missing labels never prevent synthesis.

`turn-boundaries.patch` fixes the 39.001-second task where short question/answer
pairs were merged to exceed three seconds. Every one of its seven references was
its own original file, but several files contained both voices. The v5 segmenter
forbids merging across voice-change times, even for very short replies. Complete
sentences also keep their boundaries; only adjacent clauses without a complete
sentence boundary, intervening sound event or gap over 300 ms can merge. Separate
punctuation tokens attach to preceding text without creating empty audio windows.
Scribe's native punctuation is used directly, avoiding redundant punctuation
restoration and empty timestamp-bearing tokens. Old Scribe caches without
`turn_detection=boundary_only` are refreshed; synthesis caches are version-isolated.

Every resulting utterance still supplies both IndexTTS prompts. Short conditioning
references pad only that utterance with silence; no neighboring voice is borrowed.
The final Docker layer runs 67 regressions. `repair_turn_boundaries.py JOB_ID`
reuses separated audio, re-recognizes with turn boundaries, translates and
synthesizes under the shared execution lock, then checks no segment spans a
returned turn boundary, exact aligned frame counts, reference sample provenance
and full output decoding. It writes correction artifacts separately. Activation
and `publish_segment_correction.py STUDIO_JOB_ID turn-boundaries-v1` replace only
the selected completed result, with no new task or Studio balance debit.

Validated and published the correction for Studio `job_ca4bceab5c715909` / engine
`70cbaa6b-48e4-5a3b-b290-3acd1f7bfd55`: 11 returned change times, 17 independently
referenced utterances instead of 7 merged ones, all 17 synthesized, zero fallback
warnings. This verifies boundaries and reference provenance, not a guarantee of
perceptual voice fidelity for every very short utterance. The selected result is
`result-turn-boundaries-v1.mp4` with its new VTT and embedded switchable subtitles.
No new charge was created. Earlier output artifacts are retained.
