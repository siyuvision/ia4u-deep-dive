import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_points import match_points

def test_empty():
    assert match_points([],[[0,0]],12)==[]

def test_one_to_one():
    assert len(match_points([[0,0]],[[0,0],[1,0]],12))==1

def test_radius_gate():
    assert match_points([[0,0]],[[13,0]],12)==[]

def test_maximum_cardinality_before_distance():
    assert len(match_points([[0,0],[2,0]],[[1,0],[-1,0]],1.1))==2

if __name__=='__main__':
    for test in [test_empty,test_one_to_one,test_radius_gate,test_maximum_cardinality_before_distance]: test()
    print('4 matching tests passed')
