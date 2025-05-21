import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, BarChart, Bar } from 'recharts';

const StepFunctionVisualizer = () => {
  // Constants
  const MIN_X = -0.25;
  const MAX_X = 0.25;
  const DEFAULT_NUM_PIECES = 50;
  const MAX_PIECES = 200;
  const GOOGLE_BOUND = 1.5053;
  const MAX_HEIGHT = 100; // Increased from 20

  // State variables
  const [numPieces, setNumPieces] = useState(DEFAULT_NUM_PIECES);
  const [stepFunction, setStepFunction] = useState([]);
  const [autoconvolution, setAutoconvolution] = useState([]);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [currentHeight, setCurrentHeight] = useState(0);
  const [totalHeight, setTotalHeight] = useState(0);
  const [maxAutoconvValue, setMaxAutoconvValue] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  
  // Refs
  const barChartRef = useRef(null);
  
  // Reference to store the current values for debounced operations
  const stateRef = useRef({
    autoConvTimeout: null,
    maxAutoTimeout: null,
    stepFunction: []
  });
  
  // Initialize step function on component mount or when number of pieces changes
  useEffect(() => {
    generateRandomStepFunction();
  }, [numPieces]);
  
  // Update max autoconvolution value when autoconvolution changes
  useEffect(() => {
    if (autoconvolution.length > 0) {
      const maxVal = Math.max(...autoconvolution.map(point => point.y));
      setMaxAutoconvValue(maxVal);
    }
  }, [autoconvolution]);
  
  // Calculate autoconvolution
  const calculateAutoconvolution = useCallback((steps) => {
    const P = numPieces;
    const heights = steps.map(step => step.y);
    const result = [];
    
    // Calculate integral of f(x): (1/2P) * sum of heights
    const pieceWidth = 1/(2*P); // Width of each piece is (MAX_X - MIN_X)/P = 0.5/P = 1/(2P)
    const integral = pieceWidth * heights.reduce((acc, h) => acc + h, 0);
    const integralSquared = integral * integral;
    
    // Calculate over a wider range for visualization
    for (let m = 0; m <= 2 * P; m++) {
      const t = -0.5 + m / (2 * P);
      let value = 0;
      
      // Skip computation for boundary points
      if (m === 0 || m === 2 * P) {
        value = 0;
      } else {
        const kMin = Math.max(0, m - P);
        const kMax = Math.min(P - 1, m - 1);
        
        for (let k = kMin; k <= kMax; k++) {
          value += heights[k] * heights[m - 1 - k];
        }
        
        // Multiply by piece width and divide by integral squared to normalize properly
        value = (pieceWidth * value) / integralSquared;
      }
      
      result.push({
        x: t,
        y: value
      });
    }
    
    setAutoconvolution(result);
  }, [numPieces]);
  
  // Update total height
  const updateTotalHeight = useCallback((steps) => {
    const sum = steps.reduce((acc, step) => acc + step.y, 0);
    setTotalHeight(sum);
  }, []);
  
  // Generate random step function
  const generateRandomStepFunction = useCallback(() => {
    const pieceWidth = (MAX_X - MIN_X) / numPieces;
    let newStepFunction = [];
    
    for (let i = 0; i < numPieces; i++) {
      const x = MIN_X + (i + 0.5) * pieceWidth;
      const height = Math.random() * (MAX_HEIGHT / 5); // Using MAX_HEIGHT/5 for initial values to leave room for adjustment
      
      newStepFunction.push({
        x,
        y: height,
        pieceIndex: i,
        width: pieceWidth
      });
    }
    
    setStepFunction(newStepFunction);
    setSelectedPiece(null);
    setCurrentHeight(0);
    
    calculateAutoconvolution(newStepFunction);
    updateTotalHeight(newStepFunction);
  }, [numPieces, calculateAutoconvolution, updateTotalHeight]);
  
  // Handle piece selection
  const handlePieceClick = useCallback((index) => {
    if (isDragging) return;
    
    // Find the piece at the given index
    const piece = stepFunction[index];
    
    // Update the height slider to match the selected piece's height
    setCurrentHeight(piece.y);
    
    // Set the selected piece
    setSelectedPiece(index);
  }, [stepFunction, isDragging]);
  
  // Handle piece deselection
  const handlePieceDeselect = useCallback(() => {
    setSelectedPiece(null);
    setCurrentHeight(0);
  }, []);
  
  // Handle height slider change
  const handleHeightChange = useCallback((newHeight) => {
    if (selectedPiece === null) return;
    
    // Update current height
    setCurrentHeight(newHeight);
    
    // Update step function and calculate autoconvolution immediately
    setStepFunction(prev => {
      const updated = [...prev];
      updated[selectedPiece] = {
        ...updated[selectedPiece],
        y: newHeight
      };
      
      // Calculate autoconvolution immediately for real-time feedback
      calculateAutoconvolution(updated);
      updateTotalHeight(updated);
      
      return updated;
    });
  }, [selectedPiece, calculateAutoconvolution, updateTotalHeight]);
  
  // Handle drag start
  const handleDragStart = useCallback((event, index) => {
    if (index === undefined || index === null) return;
    
    // If clicked on a different piece than currently selected, select it first
    if (selectedPiece !== index) {
      handlePieceClick(index);
      return;
    }
    
    setIsDragging(true);
    
    const chartRect = barChartRef.current.getBoundingClientRect();
    const startY = event.clientY;
    const startHeight = currentHeight;
    
    const handleMouseMove = (moveEvent) => {
      // Calculate delta from start position
      const deltaY = startY - moveEvent.clientY;
      
      // Scale delta to height (higher = more height)
      const heightScale = MAX_HEIGHT / (chartRect.height * 0.4); // Reduced divisor to allow for larger height changes
      const newHeight = Math.max(0, startHeight + deltaY * heightScale); // Removed upper limit
      
      // Update current height
      setCurrentHeight(newHeight);
      
      // Update step function and calculate autoconvolution in real-time
      setStepFunction(prev => {
        const updated = [...prev];
        updated[selectedPiece] = {
          ...updated[selectedPiece],
          y: newHeight
        };
        
        // Calculate autoconvolution immediately for real-time feedback
        calculateAutoconvolution(updated);
        updateTotalHeight(updated);
        
        return updated;
      });
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      
      // Remove event listeners
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    // Add event listeners
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [selectedPiece, currentHeight, handleHeightChange, handlePieceClick]);
  
  // Custom bar chart component
  const CustomBarChart = () => {
    return (
      <div ref={barChartRef} className="w-full h-64 relative">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            data={stepFunction} 
            barCategoryGap={0} 
            barGap={0}
            onClick={(data) => {
              if (data && data.activeTooltipIndex !== undefined) {
                handlePieceClick(data.activeTooltipIndex);
              }
            }}
            isAnimationActive={false}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              type="number"
              domain={[MIN_X, MAX_X]}
              tickCount={11}
              label={{ value: 'x', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              domain={[0, 'auto']}
              label={{ value: 'f(x)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              formatter={(value) => [value.toFixed(2), 'f(x)']}
              labelFormatter={(label) => `x: ${Number(label).toFixed(4)}`}
              isAnimationActive={false}
            />
            <ReferenceLine y={0} stroke="#000" />
            <ReferenceLine x={0} stroke="#000" />
            <Bar 
              dataKey="y" 
              isAnimationActive={false}
              shape={(props) => {
                const { x, y, width, height, index, background } = props;
                const chartHeight = background?.height || 0;
                
                return (
                  <g>
                    {/* Invisible full-height clickable area */}
                    <rect
                      x={x}
                      y={0}
                      width={width}
                      height={chartHeight}
                      fill="transparent"
                      cursor={selectedPiece === index ? 'ns-resize' : 'pointer'}
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePieceClick(index);
                      }}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        handleDragStart(e, index);
                      }}
                    />
                    
                    {/* Visible bar */}
                    <rect
                      x={x}
                      y={y}
                      width={width}
                      height={height}
                      fill={selectedPiece === index ? "#ff7300" : "#8884d8"}
                      cursor={selectedPiece === index ? 'ns-resize' : 'pointer'}
                      pointerEvents="none" // Let the invisible rectangle handle events
                    />
                  </g>
                );
              }}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };
  
  // Custom line chart for autoconvolution
  const CustomLineChart = () => {
    return (
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart 
            data={autoconvolution}
            isAnimationActive={false}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="x"
              type="number"
              domain={[-0.5, 0.5]}
              tickCount={11}
              label={{ value: 't', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              domain={[0, 'auto']}
              label={{ value: '(f*f)(t)/(∫f)²', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              formatter={(value) => [value.toFixed(4), '(f*f)(t)/(∫f)²']}
              labelFormatter={(label) => `t: ${Number(label).toFixed(4)}`}
              isAnimationActive={false}
            />
            <ReferenceLine y={0} stroke="#000" />
            <ReferenceLine x={0} stroke="#000" />
            <ReferenceLine y={GOOGLE_BOUND} stroke="#FF0000" strokeDasharray="5 5" />
            <Line
              type="monotone"
              dataKey="y"
              stroke="#82ca9d"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };
  
  return (
    <div className="flex flex-col p-4 space-y-4 max-w-5xl mx-auto">
      <div className="text-2xl font-bold text-center">Step Function Autoconvolution</div>
      
      {/* Controls */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="mb-2">Number of Pieces: {numPieces}</div>
            <input
              type="range"
              min="5"
              max={MAX_PIECES}
              value={numPieces}
              onChange={(e) => setNumPieces(Number(e.target.value))}
              className="w-full"
            />
          </div>
          
          <div>
            <button
              onClick={generateRandomStepFunction}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            >
              Regenerate Random
            </button>
          </div>
          
          <div>
            <div className="mb-2">Selected Piece: {selectedPiece !== null ? selectedPiece : 'None'}</div>
            {selectedPiece !== null && (
              <button
                onClick={handlePieceDeselect}
                className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
              >
                Deselect
              </button>
            )}
          </div>
          
          <div>
            <div className="mb-2">Height Adjustment: {currentHeight.toFixed(2)}</div>
            <div className="relative w-full">
              <input
                type="range"
                min="0"
                max={MAX_HEIGHT * 5}
                step="0.1"
                value={currentHeight}
                onInput={(e) => {
                  // For smooth updates during dragging the slider
                  const newHeight = Number(e.target.value);
                  if (selectedPiece !== null) {
                    // Update current height
                    setCurrentHeight(newHeight);
                    
                    // Update step function directly without triggering a React re-render
                    const updated = [...stepFunction];
                    updated[selectedPiece] = {
                      ...updated[selectedPiece],
                      y: newHeight
                    };
                    
                    // Calculate autoconvolution directly
                    stateRef.current.stepFunction = updated;
                    calculateAutoconvolution(updated);
                    updateTotalHeight(updated);
                  }
                }}
                onChange={(e) => {
                  // Final update when the slider is released
                  handleHeightChange(Number(e.target.value));
                }}
                className="w-full slider-thumb-orange"
                disabled={selectedPiece === null}
                style={{
                  background: selectedPiece !== null ? 
                    `linear-gradient(to right, #ff7300 0%, #ff7300 ${(currentHeight / (MAX_HEIGHT * 5)) * 100}%, #e5e7eb ${(currentHeight / (MAX_HEIGHT * 5)) * 100}%, #e5e7eb 100%)` : 
                    '#e5e7eb'
                }}
              />
            </div>
          </div>
        </div>
      </div>
      
      {/* Step Function Chart */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="font-semibold mb-2">LP Step Function (P={numPieces})</div>
        <CustomBarChart />
      </div>
      
      {/* Autoconvolution Chart */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="flex justify-between items-center mb-2">
          <div className="font-semibold">LP Autoconvolution (f*f)(t)/(∫f)² (P={numPieces})</div>
          <div className="text-green-700">Max Value: {maxAutoconvValue.toFixed(4)}</div>
        </div>
        <CustomLineChart />
        <div className="text-sm mt-2 text-red-600">Google's bound: {GOOGLE_BOUND} (red line)</div>
      </div>
      
      {/* Instructions */}
      <div className="text-sm bg-gray-100 p-4 rounded">
        <strong>Instructions:</strong>
        <ol className="list-decimal pl-5">
          <li>Click on any bar to select it (turns orange)</li>
          <li>Drag selected bar up/down to adjust height or use the slider</li>
          <li>The autoconvolution plot shows Google's bound at 1.5053</li>
        </ol>
      </div>
    </div>
  );
};

export default StepFunctionVisualizer;