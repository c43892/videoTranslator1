from videotranslator.domain import Word
from videotranslator.segmentation import segment_dialogue, parse_words

def w(text, start, end, speaker='speaker_0'):
    return Word(text, start, end, speaker)

def test_sentence_spanning_subtitle_lines_is_one_utterance():
    words = [w('I',0,.2), w('would',.2,.5), w('like\n',.5,.8), w('some',.8,1), w('tea.',1,1.5)]
    result = segment_dialogue(words)
    assert len(result) == 1
    assert result[0].source_text == 'I would like some tea.'
    assert result[0].start == 0 and result[0].end == 1.5

def test_legacy_speaker_labels_do_not_cut_unpunctuated_text():
    words = [w('Hello',0,1),w('again',2,3),w('Yes',3,4,'speaker_1')]
    result=segment_dialogue(words)
    assert len(result)==1
    assert result[0].source_text=='Hello again Yes'
    assert result[0].speaker_id==''
    assert all('mixed_speakers_in_utterance' not in s.flags for s in result)

def test_natural_clause_split_when_reference_is_overlong():
    words = [w('First,',0,8), w('second',8.1,14),w('part.',14,19)]
    result = segment_dialogue(words)
    assert len(result) == 2
    assert result[0].end == 8 and result[1].start == 8.1
    assert not any(s.flags for s in result)

def test_no_mechanical_duration_cut():
    result = segment_dialogue([w('A',0,8),w('continuous',8,16),w('utterance',16,22)])
    assert len(result) == 1
    assert result[0].end == 22
    assert not result[0].flags
    assert 'long_utterance_requires_bounded_emotion_reference' in result[0].notes

def test_chinese_spacing_and_stable_ids():
    words = [w('你',0,.3),w('好。',.3,.8),w('世界！',1,2)]
    a, b = segment_dialogue(words), segment_dialogue(words)
    assert [x.source_text for x in a] == ['你好。','世界！']
    assert 'short_utterance_no_compatible_neighbor' in a[0].notes
    assert [s.id for s in a] == [s.id for s in b]

def test_transcript_ignores_events_and_missing_timestamps():
    result = parse_words({'words':[{'type':'audio_event','text':'laughing','start':0,'end':1},
                                  {'text':'Hello','start':1,'end':2,'speaker_id':'a'},
                                  {'text':' ', 'type':'spacing'}, {'text':'invalid','start':4,'end':2}]})
    assert len(result) == 1 and result[0].speaker_id == 'a'
def test_nonfinite_timestamps_are_ignored():
    from videotranslator.segmentation import parse_words
    assert parse_words({'words':[{'text':'bad','start':float('nan'),'end':3},
                                 {'text':'bad','start':0,'end':float('inf')}]})==[]



def test_legacy_complete_sentences_then_close_subtitles_merge():
    result=segment_dialogue([w('One,',0,1),w('two.',1.1,2),w('three!',2.1,3.5),w('Tail.',3.6,4.2)])
    assert [s.source_text for s in result]==['One, two.','three!','Tail.']
    assert [(s.start,s.end) for s in result]==[(0,2),(2.1,3.5),(3.6,4.2)]


def test_complete_three_second_sentence_keeps_its_boundary():
    result=segment_dialogue([w('One.',0,3),w('Two.',4,5)])
    assert len(result)==2 and result[0].end==3


def test_provider_utterances_are_primary_editing_units_and_close_ones_merge():
    words=[w('Hello.',0,.7),w('How',1,1.3),w('are',1.3,1.5),w('you?',1.5,2),
           w('Later.',3,4)]
    utterances=[{'start':0,'end':.7,'text':'Hello.'},{'start':1,'end':2,'text':'How are you?'},
                {'start':3,'end':4,'text':'Later.'}]
    result=segment_dialogue(words,utterances=utterances)
    assert [(s.start,s.end,s.source_text) for s in result]==[(0,.7,'Hello.'),(1,2,'How are you?'),(3,4,'Later.')]


def test_provider_segment_bounds_override_inner_word_bounds():
    result=segment_dialogue([w('Hello.',1.2,1.8)],
                            utterances=[{'start':1,'end':2,'text':'Hello.'}])
    assert [(s.start,s.end) for s in result]==[(1,2)]


def test_provider_utterances_under_200ms_merge_up_to_ten_seconds():
    words=[w('One.',0,2),w('Two.',2.1,4),w('Three.',4.1,11)]
    utterances=[{'start':0,'end':2,'text':'One.'},{'start':2.1,'end':4,'text':'Two.'},
                {'start':4.1,'end':11,'text':'Three.'}]
    result=segment_dialogue(words,utterances=utterances)
    assert [(s.start,s.end,s.source_text) for s in result]==[(0,4,'One. Two.'),(4.1,11,'Three.')]


def test_short_clauses_merge_without_identity_inference():
    result=segment_dialogue([w('Hi.',0,1,'A'),w('Hello.',1.1,2,'B')])
    assert len(result)==2 and all(s.speaker_id=='' for s in result)
    assert not result[0].flags


def test_unknown_inside_japanese_word_does_not_split_or_insert_spaces():
    result=segment_dialogue([w('大',0,.4,'unknown'),w('丈',.4,.4,'A'),w('夫ですか？',.4,1,'A'),
                            w('み',1.1,1.2,'A'),w('んな',1.2,1.4,'A'),w('乗りましたね？',1.4,3.6,'A')])
    assert [s.source_text for s in result]==['大丈夫ですか？','みんな乗りましたね？']
    assert result[0].speaker_id=='' and not result[0].flags


def test_reference_length_does_not_limit_short_clause_merge():
    result=segment_dialogue([w('Long,',0,14),w('tail.',14,16)])
    assert len(result)==1 and result[0].end==16 and not result[0].flags
