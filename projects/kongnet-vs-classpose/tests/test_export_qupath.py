import json
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_qupath import export_kongnet_db, export_prediction_json


class ExportQuPathTests(unittest.TestCase):
    def test_json_preserves_coordinates_and_zero_class(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'sample.json'
            report = {'model': 'classpose', 'slide_dimensions': [100, 100],
                      'points': [{'x': 12.25, 'y': 34.5, 'class_id_conic': 0},
                                 {'x': 90, 'y': 20, 'class_id_conic': 6}]}
            source.write_text(json.dumps(report), encoding='utf-8')
            original = source.read_bytes()
            result = export_prediction_json(source)
            features = json.loads(Path(result['output']).read_text())['features']
            self.assertEqual(result['points'], 2)
            self.assertEqual(features[0]['geometry']['coordinates'], [12.25, 34.5])
            self.assertEqual(features[0]['properties']['classification']['name'],
                             'ClassPose: unclassified_0')
            self.assertEqual(source.read_bytes(), original)
            report['points'][0]['x'] = 100
            source.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(ValueError):
                export_prediction_json(source)
            self.assertEqual(json.loads(Path(result['output']).read_text())['features'], features)

    def test_db_preserves_level0_coordinates_and_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'sample.db'
            with closing(sqlite3.connect(source)) as connection:
                connection.execute('CREATE TABLE annotations (id INTEGER, cx REAL, cy REAL, objtype TEXT, properties TEXT)')
                connection.execute('INSERT INTO annotations VALUES (1,12.25,34.5,?,?)',
                                   ('Point', json.dumps({'type': 'lymphocyte', 'prob': .875})))
                connection.commit()
            original = source.read_bytes()
            result = export_kongnet_db(source)
            feature = json.loads(Path(result['output']).read_text())['features'][0]
            self.assertEqual(feature['geometry']['coordinates'], [12.25, 34.5])
            self.assertEqual(feature['properties']['measurements']['Confidence'], .875)
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
