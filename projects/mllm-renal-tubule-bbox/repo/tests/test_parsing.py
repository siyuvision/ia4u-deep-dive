from mllm_bbox.parsing import extract_boxes

QUAD = [10.0, 20.0, 30.0, 40.0]


def test_bare_array_with_bbox_2d():
    r = extract_boxes('[{"label": "renal tubule", "bbox_2d": [10, 20, 30, 40]}, {"label": "renal tubule", "bbox_2d": [1, 2, 3, 4]}]')
    assert r.mode == "json"
    assert r.boxes == [QUAD, [1.0, 2.0, 3.0, 4.0]]


def test_code_fence_with_surrounding_prose():
    text = 'Here are the boxes:\n```json\n[{"bbox_2d": [10, 20, 30, 40]}]\n```\nLet me know if you need more.'
    assert extract_boxes(text).boxes == [QUAD]


def test_accepted_shapes():
    assert extract_boxes('{"boxes": [{"bbox_2d": [10, 20, 30, 40]}]}').boxes == [QUAD]
    assert extract_boxes("[[10, 20, 30, 40]]").boxes == [QUAD]
    assert extract_boxes('[{"box_2d": [10, 20, 30, 40], "label": "x"}]').boxes == [QUAD]
    assert extract_boxes('[{"x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40}]').boxes == [QUAD]
    assert extract_boxes('[{"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40}]').boxes == [QUAD]


def test_reasoning_block_is_ignored():
    text = '<think>maybe [[1, 2, 3, 4], [5, 6, 7, 8]] is it</think>[{"bbox_2d": [10, 20, 30, 40]}]'
    assert extract_boxes(text).boxes == [QUAD]


def test_draft_then_final_prefers_the_larger_then_later_list():
    text = '[{"bbox_2d": [1, 2, 3, 4]}] revised: [{"bbox_2d": [10, 20, 30, 40]}, {"bbox_2d": [5, 6, 7, 8]}]'
    assert len(extract_boxes(text).boxes) == 2


def test_truncated_answer_keeps_every_complete_box():
    text = (
        '[{"label": "renal tubule", "bbox_2d": [1, 2, 3, 4]},'
        '{"label": "renal tubule", "bbox_2d": [5, 6, 7, 8]},'
        '{"label": "renal tubule", "bbox_2d": [9, 10'
    )
    r = extract_boxes(text)
    assert r.mode == "json_repaired"
    assert r.boxes == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_truncated_list_of_lists():
    r = extract_boxes("[[1, 2, 3, 4], [5, 6, 7")
    assert r.mode == "json_repaired" and r.boxes == [[1.0, 2.0, 3.0, 4.0]]


def test_regex_fallback_for_native_box_tokens():
    r = extract_boxes("renal tubule <|begin_of_box|>[10, 20, 30, 40]<|end_of_box|> and <|begin_of_box|>[1, 2, 3, 4]<|end_of_box|>")
    assert r.mode == "regex" and r.boxes == [QUAD, [1.0, 2.0, 3.0, 4.0]]


def test_structured_answer_wins_over_stray_quads_in_prose():
    text = 'Region [1, 2, 3, 4] looks odd. Answer: [{"bbox_2d": [10, 20, 30, 40]}]'
    r = extract_boxes(text)
    assert r.mode == "json" and r.boxes == [QUAD]


def test_nothing_found():
    for text in (None, "", "I cannot identify any tubules.", '{"count": 3}', "[1, 2, 3]"):
        r = extract_boxes(text)
        assert r.mode == "none" and r.boxes == []


def test_non_numeric_and_boolean_values_are_not_boxes():
    assert extract_boxes('[{"bbox_2d": [1, 2, 3, "x"]}]').boxes == []
    assert extract_boxes("[[true, false, true, false]]").boxes == []
