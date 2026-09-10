// Headless compatibility test on the real image; does not save/replace user data.
import qupath.lib.io.PathIO
import java.nio.file.Paths

assert args.size() == 2 : 'Arguments: GeoJSON path, expected object count'
def objects = PathIO.readObjects(Paths.get(args[0]))
assert objects.size() == Integer.parseInt(args[1])
def server = getCurrentServer()
assert server != null
objects.each { obj ->
    assert obj.isDetection()
    assert obj.getROI().isPoint()
    assert obj.getPathClass() != null
    assert obj.getROI().getCentroidX() >= 0 && obj.getROI().getCentroidX() < server.getWidth()
    assert obj.getROI().getCentroidY() >= 0 && obj.getROI().getCentroidY() < server.getHeight()
}
addObjects(objects)
assert getDetectionObjects().size() == objects.size()
println 'QUPATH_IMPORT_OK ' + objects.size() + ' objects; image=' + server.getWidth() + 'x' + server.getHeight()
println objects.countBy { it.getPathClass().toString() }
