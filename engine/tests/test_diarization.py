import httpx
import pytest
from videotranslator import diarization as d, media
from videotranslator.config import Settings
from videotranslator.domain import ProviderError, Cancelled
from videotranslator.transcription import OpenAISpeakerDiarizer


def turn(speaker, start, end):
    return {'speaker': speaker, 'start': start, 'end': end, 'text': 'Speech.'}


def test_labels_are_not_identity_and_references_preserve_returning_speaker():
    known = set()
    mapping, issues = d.identity_map([turn('A',0,3), turn('B',4,7)], [], {}, known)
    assert mapping == {'A':'speaker_001', 'B':'speaker_002'} and not issues
    refs = dict.fromkeys(known, 'reference')
    mapping, issues = d.identity_map([turn('speaker_002', 600, 603), turn('A',604,607)], [], refs, known)
    assert mapping == {'speaker_002':'speaker_002', 'A':'speaker_003'}


def test_overlap_recovers_label_and_conflicts_are_unknown():
    known = {'speaker_001', 'speaker_002'}
    previous = [turn('speaker_001',590,603)]
    mapping, _ = d.identity_map([turn('B',595,606)], previous, {}, known)
    assert mapping['B'] == 'speaker_001'
    mapping, warnings = d.identity_map([turn('speaker_002',595,606)], previous,
                                      dict.fromkeys(known,'ref'), known)
    assert mapping['speaker_002'] == 'unknown' and warnings


def test_unreferenced_returning_people_do_not_get_guessed():
    known = {f'speaker_{i:03d}' for i in range(1,6)}
    mapping, warnings = d.identity_map([turn('A',600,603)], [],
                                      dict.fromkeys(sorted(known)[:4],'ref'), known)
    assert mapping['A'] == 'unknown' and warnings


def test_chunk_boundaries_offsets_reference_upload_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(d,'SINGLE_REQUEST_SECONDS',12)
    monkeypatch.setattr(d,'CHUNK_SECONDS',10)
    monkeypatch.setattr(d,'OVERLAP_SECONDS',2)
    audio = tmp_path/'dialogue.mp3'
    media.ffmpeg(['-f','lavfi','-i','sine=duration=25', '-ar','16000','-ac','1','-b:a','48k',audio])
    calls = []
    def request(path, references):
        calls.append(path.name)
        duration = float(media.probe(path)['format']['duration'])
        assert duration < 15
        if len(calls) == 1:
            assert references == {}
            return {'segments':[turn('A',0,12)]}
        assert list(references) == ['speaker_001']
        assert references['speaker_001'].startswith('data:audio/wav;base64,')
        if len(calls) == 2:
            raise ProviderError('temporary failure')
        return {'segments':[turn('speaker_001',0,14 if path.name=='chunk-001.mp3' else 7)]}
    with pytest.raises(ProviderError):
        d.chunked_diarize(audio, request)
    result = d.chunked_diarize(audio, request)
    assert calls == ['chunk-000.mp3','chunk-001.mp3','chunk-001.mp3','chunk-002.mp3']
    assert [(t['start'], t['end']) for t in result['segments']] == [(0,10),(10,20),(20,25)]
    assert {t['speaker'] for t in result['segments']} == {'speaker_001'}
    assert d.chunked_diarize(audio, request) == result
    assert len(calls) == 4
    def cancel():
        raise Cancelled()
    with pytest.raises(Cancelled):
        d.chunked_diarize(audio, request, cancel)
    assert len(calls) == 4


def test_reference_fields_are_sent_as_multipart_arrays(tmp_path, monkeypatch):
    path = tmp_path/'dialogue.mp3'
    path.write_bytes(b'fixture')
    original = httpx.Client
    def handle(request):
        body = request.read()
        assert b'name="known_speaker_names[]"' in body and b'speaker_001' in body
        assert b'name="known_speaker_references[]"' in body and b'data:audio/wav;base64,YQ==' in body
        return httpx.Response(200,json={'segments':[]})
    monkeypatch.setattr(httpx,'Client',lambda **kw: original(transport=httpx.MockTransport(handle),**kw))
    OpenAISpeakerDiarizer(Settings()).request(path, {'speaker_001':'data:audio/wav;base64,YQ=='})


@pytest.mark.parametrize('segment',[turn('A',0,float('nan')), turn('A',0,20),turn('',0,1)])
def test_bad_response_timestamps_or_identity_are_rejected(segment):
    with pytest.raises(ProviderError):
        d.validated_turns({'segments':[segment]},10)
