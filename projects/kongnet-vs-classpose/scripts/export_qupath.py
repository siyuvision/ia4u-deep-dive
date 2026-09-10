"""Export existing level-0 detection points as QuPath GeoJSON (no inference).

Input contracts: KongNet SQLite AnnotationStore points are already in level-0
pixels; our prediction JSON stores image/level-0 pixels. Do not rescale again.
These are point detections, never inferred segmentation boundaries.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sqlite3

CLASSES = ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']
COLORS = [(230, 159, 0), (213, 94, 0), (0, 114, 178),
          (204, 121, 167), (240, 228, 66), (0, 158, 115)]


def write_points(rows, output, model, dimensions=None):
    """Rows: (x, y, class_name, confidence). Preserve each source point."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    # Only publish the completed file; a failed conversion leaves prior output intact.
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write('{"type":"FeatureCollection","features":[')
        for index, (x, y, name, confidence) in enumerate(rows):
            x, y = float(x), float(y)
            if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
                raise ValueError(f'Invalid source coordinate: {x}, {y}')
            if dimensions and (x >= dimensions[0] or y >= dimensions[1]):
                raise ValueError(f'Point outside source image: {x}, {y}')
            if name not in CLASSES and name != 'unclassified_0':
                raise ValueError(f'Unsupported class: {name}')
            color = COLORS[CLASSES.index(name)] if name in CLASSES else (128, 128, 128)
            properties = {
                'objectType': 'detection',
                'classification': {'name': f'{model}: {name}', 'color': list(color)},
                'name': f'{model} point {index + 1}',
            }
            if confidence is not None:
                confidence = float(confidence)
                if not math.isfinite(confidence):
                    raise ValueError('Non-finite confidence')
                properties['measurements'] = {'Confidence': confidence}
            feature = {'type': 'Feature',
                       'geometry': {'type': 'Point', 'coordinates': [x, y]},
                       'properties': properties}
            if index:
                stream.write(',')
            stream.write(json.dumps(feature, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')))
            counts[name] += 1
        stream.write(']}\n')
    temporary.replace(output)
    return {'output': str(output), 'points': sum(counts.values()),
            'classes': dict(counts), 'geometry': 'Point'}


def export_prediction_json(source, output=None):
    source = Path(source)
    report = json.loads(source.read_text(encoding='utf-8'))
    model = {'kongnet': 'KongNet', 'classpose': 'ClassPose'}[report['model']]
    dimensions = report.get('slide_dimensions')
    if dimensions is None and 'image_shape' in report:
        dimensions = report['image_shape'][1], report['image_shape'][0]

    def rows():
        for point in report['points']:
            # Refuse legacy ambiguous labels instead of silently guessing offsets.
            label = point['class_id_conic']
            if label != int(label) or not 0 <= label <= 6:
                raise ValueError(f'Invalid CoNIC label: {label}')
            name = CLASSES[int(label) - 1] if label else 'unclassified_0'
            yield point['x'], point['y'], name, point.get('score')

    return write_points(rows(), output or source.with_suffix('.qupath.geojson'),
                        model, dimensions)


def export_kongnet_db(source, output=None):
    source = Path(source).resolve()
    connection = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
    try:
        def rows():
            for x, y, objtype, properties in connection.execute(
                    'SELECT cx,cy,objtype,properties FROM annotations ORDER BY id'):
                if objtype != 'Point':
                    raise ValueError('This exporter only supports saved point detections')
                properties = json.loads(properties)
                yield x, y, properties['type'], properties.get('prob')
        return write_points(rows(), output or source.with_suffix('.qupath.geojson'), 'KongNet')
    finally:
        connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export = export_kongnet_db if args.source.suffix.lower() == '.db' else export_prediction_json
    print(json.dumps(export(args.source, args.output), ensure_ascii=False))
