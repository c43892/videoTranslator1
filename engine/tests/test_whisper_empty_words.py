import pytest
from videotranslator.transcription import attach_speakers
from videotranslator.domain import ProviderError


def test_empty_alignment_entries_preserve_actual_words_and_timestamps():
    transcript = {'text':'Hello world.', 'words':[
        {'word':'Hello','start':0,'end':1}, {'word':'','start':1,'end':1.2},
        {'word':'world','start':1.2,'end':2}, {'word':' ','start':2,'end':2}]}
    result = attach_speakers(transcript, {'segments':[{'start':0,'end':2,'speaker':'A'}]}, 2)
    assert [(w['text'],w['start'],w['end'],w['speaker_id']) for w in result['words']] == [
        ('Hello',0,1,'A'),('world.',1.2,2,'A')]
    assert len(transcript['words']) == 4


def test_missing_real_word_timestamps_still_fails():
    with pytest.raises(ProviderError):
        attach_speakers({'text':'Speech','words':[{'word':'','start':0,'end':1}]}, {'segments':[]}, 1)
