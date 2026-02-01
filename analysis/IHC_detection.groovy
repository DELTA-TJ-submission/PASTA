// QuPath IHC Tile Analysis Pipeline
// Function: Divide IHC images into tiles of specified size, filter tiles with high positive cell ratio and perform statistical analysis

import qupath.lib.objects.PathObjects
import qupath.lib.roi.RectangleROI
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

// ===========================================
// Parameter Settings (adjust according to different biomarkers)
// ===========================================

// Tile size setting
def tileSize = 3000  // pixels

// Positive cell ratio threshold 
def positiveThreshold = 0.0

// DAB positive detection threshold
def dabThreshold = 0.15

// Minimum cell count threshold (to avoid false positives in low cell density areas)
def minCellsPerTile = 50

// ===========================================
// Main Pipeline
// ===========================================

def imageData = getCurrentImageData()
if (imageData == null) {
    println "Error: No image data is open"
    return
}

def server = imageData.getServer()

// Get image dimensions
def imageWidth = server.getWidth()
def imageHeight = server.getHeight()

println "Image dimensions: ${imageWidth} x ${imageHeight}"
println "Starting to create ${tileSize}x${tileSize} pixel tiles..."

// Clear existing objects
clearAllObjects()

def validTiles = []
def tileResults = []

// ===========================================
// Step 1: Create tiles
// ===========================================

def allTiles = []
def tileCount = 0

for (int x = 0; x < imageWidth; x += tileSize) {
    for (int y = 0; y < imageHeight; y += tileSize) {
        tileCount++
        
        // Calculate actual size of current tile (handle boundary cases)
        def actualWidth = Math.min(tileSize, imageWidth - x)
        def actualHeight = Math.min(tileSize, imageHeight - y)
        
        // Skip tiles that are too small at boundaries
        if (actualWidth < tileSize * 0.5 || actualHeight < tileSize * 0.5) {
            continue
        }
        
        // Create tile ROI
        def roi = new RectangleROI(x, y, actualWidth, actualHeight, ImagePlane.getDefaultPlane())
        def tile = PathObjects.createAnnotationObject(roi)
        tile.setName("Tile_${x}_${y}")
        
        allTiles.add([tile: tile, x: x, y: y, width: actualWidth, height: actualHeight])
    }
}

// Add all tiles to the image
allTiles.each { tileInfo ->
    addObject(tileInfo.tile)
}

println "Created ${allTiles.size()} tiles"

// ===========================================
// Step 2: Batch cell detection
// ===========================================

// Select all tiles
selectObjects { it.isAnnotation() }

println "Starting batch cell detection..."

// Check if color deconvolution is already set
def stains = imageData.getColorDeconvolutionStains()
if (stains == null) {
    println "Warning: Color deconvolution not detected, will attempt to use default H-DAB settings"
    // Set default H-DAB color deconvolution
    try {
        setColorDeconvolutionStains('{"Name": "H-DAB default", "Stain 1": "Hematoxylin", "Values 1": "0.65111 0.70119 0.29049", "Stain 2": "DAB", "Values 2": "0.26917 0.56824 0.77759", "Background": " 255 255 255"}')
        println "Default H-DAB color deconvolution has been set"
    } catch (Exception e) {
        println "Failed to set color deconvolution: ${e.getMessage()}"
    }
}

// Use QuPath built-in function for cell detection
try {
    // Cell detection - corrected parameter names
    def cellDetectionParams = [
        'detectionImage': 'Hematoxylin OD',
        'requestedPixelSizeMicrons': 0.5,
        'backgroundRadiusMicrons': 6.0,
        'medianRadiusMicrons': 0,
        'sigmaMicrons': 1.0,
        'minAreaMicrons': 10.0,
        'maxAreaMicrons': 400.0,
        'threshold': 0.1,
        'maxBackground': 2.0,
        'watershedPostProcess': true,
        'cellExpansionMicrons': 4.0,
        'includeNuclei': true,
        'smoothBoundaries': true,
        'makeMeasurements': true
    ]
    
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', cellDetectionParams)
    println "Cell detection completed, starting positive classification..."
    
    // Positive cell classification - corrected parameters
    def positiveCellParams = [
        'detectionImageBrightfield': 'Hematoxylin OD', // 'Optical density sum',
        'detectionImageIHC': 'DAB OD',
        'requestedPixelSizeMicrons': 0.5,
        'backgroundRadiusMicrons': 6.0,
        'medianRadiusMicrons': 0,
        'sigmaMicrons': 1.0,
        'minAreaMicrons': 10.0,
        'maxAreaMicrons': 400.0,
        'threshold': 0.1,
        'maxBackground': 2.0,
        'watershedPostProcess': true,
        'excludeDAB': false,
        'cellExpansionMicrons': 4.0,
        'includeNuclei': true,
        'smoothBoundaries': true,
        'makeMeasurements': true,
        'thresholdCompartment': 'Cell: DAB OD mean',  //change according to the biomarker
        'thresholdPositive1': dabThreshold,
        'thresholdPositive2': dabThreshold * 2,
        'thresholdPositive3': dabThreshold * 3,
        'singleThreshold': true
    ]
    
    runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', positiveCellParams)
    println "Positive cell classification completed"
    
} catch (Exception e) {
    println "Detection plugin failed: ${e.getMessage()}"
    println "Error: ${e.getStackTrace()}"
    println "Please check:"
    println "1. Whether color deconvolution is correctly set"
    println "2. Whether detection parameters are suitable for the current image"
    println "3. Whether the QuPath version supports the plugins used"
    return
}

// Wait for detection to complete
Thread.sleep(1000)

// ===========================================
// Step 3: Analyze each tile
// ===========================================

def validTileCount = 0
def processedTileCount = 0

allTiles.each { tileInfo ->
    processedTileCount++
    def tile = tileInfo.tile
    def x = tileInfo.x
    def y = tileInfo.y
    def actualWidth = tileInfo.width
    def actualHeight = tileInfo.height
    
    // Get all cell detection results in this tile
    def detections = getDetectionObjects().findAll { detection ->
        def detectionROI = detection.getROI()
        def tileROI = tile.getROI()
        
        // Check if detection is within the current tile
        return tileROI.contains(detectionROI.getCentroidX(), detectionROI.getCentroidY())
    }
    
    def totalCells = detections.size()
    
    // Skip tiles with no cells or too few cells
    if (totalCells < minCellsPerTile) {
        removeObject(tile, false)
        return
    }
    
    // ===========================================
    // Calculate positive cell statistics
    // ===========================================
    
    def positiveCells = 0
    def negativeCells = 0
    def unknownCells = 0
    
    // Count positive cells - improved classification logic
    detections.each { detection ->
        def pathClass = detection.getPathClass()
        def measurements = detection.getMeasurementList()
        def isPositive = false
        def isClassified = false
        
        // Method 1: Based on PathClass classification (more precise matching)
        if (pathClass != null) {
            def className = pathClass.getName()
            if (className != null) {
                def lowerClassName = className.toLowerCase()
                if (lowerClassName.contains("positive") || lowerClassName.contains("pos") || 
                    lowerClassName.contains("1+") || lowerClassName.contains("2+") || 
                    lowerClassName.contains("3+") || lowerClassName.contains("tumor")) {
                    isPositive = true
                    isClassified = true
                } else if (lowerClassName.contains("negative") || lowerClassName.contains("neg")) {
                    isPositive = false
                    isClassified = true
                }
            }
        }
        
        // Method 2: Based on DAB intensity measurements (when PathClass classification is unclear)
        if (!isClassified) {
            def dabMeasurements = [
                "Nucleus: DAB OD mean",
                "Cytoplasm: DAB OD mean", 
                "Cell: DAB OD mean",
                "DAB OD mean"
            ]
            
            for (measurementName in dabMeasurements) {
                try {
                    if (measurements.containsNamedMeasurement(measurementName)) {
                        def dabIntensity = measurements.getMeasurementValue(measurementName)
                        if (!Double.isNaN(dabIntensity) && dabIntensity > dabThreshold) {
                            isPositive = true
                            isClassified = true
                            break
                        } else if (!Double.isNaN(dabIntensity)) {
                            isPositive = false
                            isClassified = true
                        }
                    }
                } catch (Exception e) {
                    // Ignore measurement reading errors
                }
            }
        }
        
        // Collect statistics
        if (isClassified) {
            if (isPositive) {
                positiveCells++
            } else {
                negativeCells++
            }
        } else {
            unknownCells++
        }
    }
    
    // Calculate positive ratio (only considering classified cells)
    def classifiedCells = positiveCells + negativeCells
    def positiveRatio = classifiedCells > 0 ? (double)positiveCells / classifiedCells : 0.0
    
    // ===========================================
    // Filter tiles that meet the criteria
    // ===========================================
    
    if (positiveRatio >= positiveThreshold && classifiedCells >= minCellsPerTile) {
        validTileCount++
        
        // Set tile classification and color
        tile.setPathClass(getPathClass("Valid_Tile", makeRGB(255, 100, 100)))
        
        // Set color intensity based on positive ratio
        def intensity = Math.min(200, (int)(positiveRatio * 200 / 0.8))
        def color = makeRGB(255, 255 - intensity, 255 - intensity)
        tile.setColorRGB(color)
        
        // Record tile information
        def tileInfo_result = [
            name: "Tile_${x}_${y}",
            x: x,
            y: y,
            width: actualWidth,
            height: actualHeight,
            totalCells: totalCells,
            classifiedCells: classifiedCells,
            positiveCells: positiveCells,
            negativeCells: negativeCells,
            unknownCells: unknownCells,
            positiveRatio: positiveRatio
        ]
        
        tileResults.add(tileInfo_result)
        validTiles.add(tile)
        
        // Add measurement results to tile
        def measurements = tile.getMeasurementList()
        measurements.putMeasurement("Total Cells", totalCells)
        measurements.putMeasurement("Classified Cells", classifiedCells)
        measurements.putMeasurement("Positive Cells", positiveCells)
        measurements.putMeasurement("Negative Cells", negativeCells)
        measurements.putMeasurement("Unknown Cells", unknownCells)
        measurements.putMeasurement("Positive Ratio", positiveRatio)
        measurements.close()
        
    } else {
        // Remove tiles that don't meet the criteria
        removeObject(tile, false)
    }
}

// ===========================================
// Step 4: Output statistical results
// ===========================================

println "\n" + "="*60
println "IHC Tile Analysis Completed!"
println "="*60

println "Overall Statistics:"
println "- Total tiles: ${allTiles.size()}"
println "- Processed tiles: ${processedTileCount}"
println "- Valid tiles: ${validTileCount}"
if (allTiles.size() > 0) {
    println "- Selection rate: ${String.format('%.2f%%', (validTileCount / allTiles.size()) * 100)}"
}

if (validTileCount > 0) {
    // Calculate overall statistics
    def totalCellsAll = tileResults.sum { it.totalCells }
    def classifiedCellsAll = tileResults.sum { it.classifiedCells }
    def positiveCellsAll = tileResults.sum { it.positiveCells }
    def negativeCellsAll = tileResults.sum { it.negativeCells }
    def unknownCellsAll = tileResults.sum { it.unknownCells }
    
    def avgPositiveRatio = tileResults.collect { it.positiveRatio }.sum() / validTileCount
    def maxPositiveRatio = tileResults.collect { it.positiveRatio }.max()
    def minPositiveRatio = tileResults.collect { it.positiveRatio }.min()
    
    println "\nStatistics for Valid Tiles:"
    println "- Total cells: ${totalCellsAll}"
    println "- Classified cells: ${classifiedCellsAll}"
    println "- Positive cells: ${positiveCellsAll}"
    println "- Negative cells: ${negativeCellsAll}"
    println "- Unknown cells: ${unknownCellsAll}"
    println "- Overall positive ratio: ${String.format('%.4f (%.2f%%)', (double)positiveCellsAll/classifiedCellsAll, (double)positiveCellsAll/classifiedCellsAll * 100)}"
    println "- Average positive ratio: ${String.format('%.4f (%.2f%%)', avgPositiveRatio, avgPositiveRatio * 100)}"
    println "- Maximum positive ratio: ${String.format('%.4f (%.2f%%)', maxPositiveRatio, maxPositiveRatio * 100)}"
    println "- Minimum positive ratio: ${String.format('%.4f (%.2f%%)', minPositiveRatio, minPositiveRatio * 100)}"
    
    println "\nDetailed Results (sorted by positive ratio, showing top 15):"
    println "No.\tTile Name\t\tPosition(x,y)\t\tTotal Cells\tClassified\tPositive\tPositive Ratio(%)"
    println "-" * 100
    
    // Sort by positive ratio in descending order
    def sortedResults = tileResults.sort { -it.positiveRatio }
    
    sortedResults.take(15).eachWithIndex { tile, index ->
        println "${index+1}\t${tile.name}\t(${tile.x}, ${tile.y})\t\t${tile.totalCells}\t${tile.classifiedCells}\t\t${tile.positiveCells}\t\t${String.format('%.2f', tile.positiveRatio * 100)}"
    }
    
    if (validTileCount > 15) {
        println "... ${validTileCount - 15} more tiles"
    }
    
    // ===========================================
    // Step 5: Export results to CSV file
    // ===========================================
    
    def timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
    def imageName = "IHC_Analysis"
    
    // Safely get image name
    try {
        def entry = getProjectEntry()
        if (entry != null) {
            imageName = entry.getImageName()
            // Clean special characters in filename
            imageName = imageName.replaceAll('[^a-zA-Z0-9._-]', '_')
        }
    } catch (Exception e) {
        imageName = "IHC_Analysis_${timestamp}"
    }
    
    // Determine output directory
    def outputDir = new File(System.getProperty("user.home"), "QuPath_Results")
    if (!outputDir.exists()) {
        outputDir.mkdirs()
    }
    
    def fileName = "${imageName}_tile_analysis_${timestamp}.csv"
    def csvFile = new File(outputDir, fileName)
    
    try {
        csvFile.withWriter('UTF-8') { writer ->
            writer.writeLine("Tile_Name,X_Position,Y_Position,Width,Height,Total_Cells,Classified_Cells,Positive_Cells,Negative_Cells,Unknown_Cells,Positive_Ratio,Positive_Percentage")
            sortedResults.each { tile ->
                writer.writeLine("${tile.name},${tile.x},${tile.y},${tile.width},${tile.height},${tile.totalCells},${tile.classifiedCells},${tile.positiveCells},${tile.negativeCells},${tile.unknownCells},${String.format('%.6f', tile.positiveRatio)},${String.format('%.2f', tile.positiveRatio * 100)}")
            }
        }
        println "\nResults exported to: ${csvFile.getAbsolutePath()}"
    } catch (Exception e) {
        println "CSV export failed: ${e.getMessage()}"
    }
    
    // ===========================================
    // Step 6: Create statistical summary
    // ===========================================
    
    def summaryFile = new File(outputDir, "${imageName}_analysis_summary_${timestamp}.txt")
    try {
        summaryFile.withWriter('UTF-8') { writer ->
            writer.writeLine("IHC Tile Analysis Summary")
            writer.writeLine("=" * 50)
            writer.writeLine("Image: ${imageName}")
            writer.writeLine("Analysis Time: ${new Date()}")
            writer.writeLine("Tile Size: ${tileSize} x ${tileSize} pixels")
            writer.writeLine("Positive Threshold: ${positiveThreshold} (${positiveThreshold * 100}%)")
            writer.writeLine("DAB Threshold: ${dabThreshold}")
            writer.writeLine("Min Cells Per Tile: ${minCellsPerTile}")
            writer.writeLine("")
            writer.writeLine("Results:")
            writer.writeLine("- Total Tiles: ${allTiles.size()}")
            writer.writeLine("- Valid Tiles: ${validTileCount}")
            writer.writeLine("- Selection Rate: ${String.format('%.2f%%', validTileCount > 0 ? (validTileCount / allTiles.size()) * 100 : 0)}")
            writer.writeLine("- Total Cells: ${totalCellsAll}")
            writer.writeLine("- Classified Cells: ${classifiedCellsAll}")
            writer.writeLine("- Positive Cells: ${positiveCellsAll}")
            writer.writeLine("- Classification Rate: ${String.format('%.2f%%', (double)classifiedCellsAll/totalCellsAll * 100)}")
            writer.writeLine("- Overall Positive Rate: ${String.format('%.2f%%', classifiedCellsAll > 0 ? (double)positiveCellsAll/classifiedCellsAll * 100 : 0)}")
            writer.writeLine("- Average Positive Rate: ${String.format('%.2f%%', avgPositiveRatio * 100)}")
            writer.writeLine("- Max Positive Rate: ${String.format('%.2f%%', maxPositiveRatio * 100)}")
            writer.writeLine("- Min Positive Rate: ${String.format('%.2f%%', minPositiveRatio * 100)}")
        }
        println "Analysis summary saved to: ${summaryFile.getAbsolutePath()}"
    } catch (Exception e) {
        println "Summary file save failed: ${e.getMessage()}"
    }
    
} else {
    println "\nNo tiles meeting the criteria were found!"
    println "Suggestions:"
    println "1. Lower the positive cell ratio threshold (current: ${positiveThreshold * 100}%)"
    println "2. Adjust the DAB intensity threshold (current: ${dabThreshold})"
    println "3. Lower the minimum cell count threshold (current: ${minCellsPerTile})"
    println "4. Check if cell detection parameters are suitable for the current image"
    println "5. Confirm that color deconvolution settings are correct"
}

// Refresh display
fireHierarchyUpdate()

println "\nAnalysis completed! Valid tiles are highlighted in red on the image."
println "Color intensity indicates the level of positive cell ratio."