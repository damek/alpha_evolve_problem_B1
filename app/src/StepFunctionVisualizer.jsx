import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, BarChart, Bar } from 'recharts';

const StepFunctionVisualizer = () => {
  // Constants
  const MIN_X = -0.25;
  const MAX_X = 0.25;
  const DEFAULT_NUM_PIECES = 50;
  const MAX_PIECES = 600; // Increased to accommodate Google's solution
  const GOOGLE_BOUND = 1.5053;
  const MAX_HEIGHT = 100; // Increased from 20
  
  // Predefined step functions
  const GOOGLE_SOLUTION = [9.00854017233134, 4.581788824134047, 5.954983797866223, 3.7314786029786324, 4.25817159660483, 3.544987049547799, 0.08876194700959494, 0.0491697316439318, 1.7439263266999894, 3.8548164683263795, 3.621038728073569, 4.04587668218835, 4.68211946529981, 5.5904236142896675, 4.737832747546433, 3.1594093823451055, 1.902874129984629, 2.7870307391136304, 3.277574995692391, 1.8981329099596054, 1.526040859367755, 2.305128838504833, 5.17673786436095, 4.583218228762042, 3.9910761392791887, 2.784600928752006, 5.450687602543662, 6.170368723277989, 7.045569321986071e-16, 7.149948549556939e-15, 0.0, 0.0, 0.0, 0.0, 1.2580295353835013e-15, 0.0, 0.0, 0.0, 0.0, 3.873037303627252e-15, 0.0, 0.0, 2.020385656008168e-06, 0.000293922119342568, 0.0, 4.9514916125368726e-15, 7.282654612521097e-16, 1.906059354629418e-14, 0.0, 3.3528418595404916e-15, 1.5099558045700925e-15, 4.901439953827422e-15, 0.0, 8.851999542886555e-15, 0.0, 0.0, 0.0005211322699854395, 0.3757576289315001, 0.25176470069965495, 4.1179587840945515e-06, 0.0, 2.946431316197597e-15, 0.0, 1.0333089131925899e-16, 2.591940622467849e-15, 0.0, 6.852171628124262e-15, 0.0, 0.0, 1.3885601200927435e-14, 2.5015636739088256e-15, 1.4382184696274247e-14, 1.235388698636516e-15, 9.328196456283097e-15, 6.938490364750181e-15, 5.581796597296351e-17, 0.0, 0.0, 5.1220388613389905e-15, 0.0, 6.085199919293191e-15, 0.0, 0.0, 1.0633201915504476e-14, 6.240893078396387e-16, 0.0, 9.242385301100576e-15, 2.1818685641605435e-15, 0.0, 3.841626602268906e-15, 0.0013592097228050644, 8.120066974555713e-15, 8.479388423870961e-16, 2.5924005380166956e-15, 0.0, 2.6610672065525727e-15, 0.0, 1.233819156251431e-14, 8.819083406210366e-15, 0.0, 4.492323424835768e-16, 0.0, 3.0916450306058138e-15, 0.0, 0.0, 3.404186949211756e-15, 4.54126650881379e-15, 1.462631558763152e-14, 0.0, 0.0, 0.0, 1.4460597710909072e-15, 9.521734973996671e-15, 0.0, 4.559858799705722e-15, 7.864867909828807e-16, 0.0, 1.7856864350178655e-16, 0.00021045010164189585, 0.26541232693216404, 0.8094426381528257, 0.5750041584597478, 0.23313281323505236, 3.6007277514467585e-05, 0.0, 0.7828826491881691, 0.43382874037802, 1.3263698571911402, 0.5441713262465393, 0.9864380574571914, 0.6776516652004773, 0.5910950602641856, 0.507419190418916, 0.5231329501406576, 0.9391246115133585, 0.4508771959372286, 0.28283039994676146, 1.2889986480406397, 0.9649046182943108, 1.4104382244415803, 1.3916682358533747, 0.8743196646011149, 0.7627485335443527, 0.2103862254578538, 0.14545209168646947, 0.019762475547189184, 1.2279396984729254, 0.012006361768949678, 1.7677675926679783, 0.9303739918691369, 1.0966313889580412, 0.40142701455261154, 0.1477985748190306, 0.1310850821272394, 0.0027642064206369592, 0.6718883532064702, 0.287789791442545, 1.1886491680958895, 0.6459736548490735, 0.88966666001013, 0.36931312374260505, 0.6840914190936884, 0.38692129734520775, 0.8050006872194091, 0.26610729268169875, 0.002941709304056364, 0.5150673486621109, 0.4049854152265144, 1.1607178193685956, 1.7547854228356075, 0.0, 0.8531817250969695, 2.3845552035650363e-05, 0.035208188035124974, 0.06799207369201249, 0.14050016250524128, 0.4862562534194792, 1.508781726996261, 0.46943710673489225, 0.22962993226722195, 1.589825945710927e-11, 3.51517770993058e-15, 2.4398590319680178e-15, 1.1666504235544564e-06, 0.0021946672216711, 0.34171503722540436, 0.4703022197366691, 0.1313974666218601, 0.11754826815054241, 0.0, 2.2387234387833643e-16, 0.0, 7.192783695625604e-05, 0.4486935802226264, 1.234691190028419, 2.8985055264499153, 1.0234017394012231, 2.7375379465420373, 0.5899927642043619, 1.4461499611411766, 0.7033498408537826, 1.6505029216035125, 0.9593634797752735, 0.009302210703764222, 0.0004181359389419785, 0.0, 0.0, 0.0023430720926212976, 0.42801036705183393, 0.6031743194865573, 1.8862845950884395, 1.0944504439060767, 1.3978223736063145, 0.13603422891356853, 0.8568768273359568, 0.5287328963079988, 0.04201038853661816, 0.5746932650501643, 0.7698787794362285, 2.2478052766496255, 1.3267115762262056, 1.3819155415467284, 1.210307904386098, 1.2050374056121944, 0.973960636675429, 0.13506178694552, 0.0017211602091930576, 1.2080793667302383, 0.9431703684918005, 0.004927152124127672, 0.26457949335968395, 0.219096730428291, 0.8972094379125464, 1.009247390062118, 2.5396761105116816, 2.0567929964131704, 2.5384945885180765, 2.051772820060434, 2.841483226472209, 2.5484575236736253, 2.900405077014117, 2.7293223781158513, 2.8016507480694623, 2.5235338506952227, 2.842495616436774, 3.6113040879253218, 2.4409992918997654, 2.8613737519007785, 2.0376653653073236, 2.873716631081072, 2.7431139992026585, 2.3176851657187343, 2.963845077577065, 2.1297112056154828, 3.1281786712157276, 1.559962066888169, 1.5175735153572592, 1.8986372289826554, 2.422172211485286, 1.4024751115172904, 1.6645681102200025, 1.0890488631004245, 0.9551468779062758, 0.4210663124027455, 0.7844656815643463, 1.3849725648239561, 1.1400002207678432, 1.2589535564861496, 0.00010847583255872839, 0.33022246693439483, 0.009991411612394792, 3.897603693807049e-14, 0.0, 0.0, 4.615098985648224e-16, 0.0, 0.0, 0.00019552451607645426, 7.535959259635103e-15, 0.0, 0.0, 0.0, 3.391322478636254e-15, 0.0, 5.23262076392476e-16, 0.0, 0.0013138779575907105, 3.8754445450268335e-05, 0.6732744675805906, 0.1403000505612687, 0.1066587972153134, 3.248799681911591e-05, 0.0002955183459383166, 0.003362359949825627, 1.258993392593319, 0.6721840514813638, 1.4023077519116312, 0.26971646568749497, 1.5317811712427716e-06, 0.0, 0.0004643255998670316, 0.21977013225378167, 0.4192480303816599, 1.5193826730071023, 1.140986767032434e-14, 0.00023172541086911792, 0.0, 2.8126101180648514e-16, 0.3250641832764737, 0.2631639184319745, 9.581064115586714e-06, 6.911241239536039e-08, 2.1977413193942923e-15, 0.0, 0.009284626840990289, 0.38181362260215157, 0.9229241174303233, 1.2389265199825776, 4.003701357117134e-15, 0.0018347819223694948, 0.09741977675536181, 0.7177732728524777, 0.893918190086622, 0.7592094697415344, 0.33941833076021116, 0.9895237780462212, 1.2103201657651075, 0.1562357060123791, 0.05779942219135546, 0.046035988201598356, 0.20605598834789451, 0.7087789468745198, 0.9835248618606901, 0.15343068950958472, 0.7458575667928422, 1.1650897270704061, 0.8603305066335794, 0.21035575411632462, 0.1359607526732069, 0.27083984880209855, 0.6401471037748572, 3.021195033008772e-11, 0.2302514662496976, 0.6133656015904811, 0.0014526177283533382, 0.2833008515927792, 0.6173666814379335, 0.35174441102605253, 0.42994053621927014, 0.17019592622467353, 0.18730260272562482, 0.20732692125660646, 0.0006488239072035948, 0.0008689690193708769, 0.0060827028748053225, 0.42803050608386095, 0.5802023431307022, 0.07735146147149399, 0.2677857202504946, 0.0009184606747345728, 0.43037124020293993, 0.3617534843366182, 0.3772422594534307, 0.2584947755294139, 0.001169636807952639, 1.9643151751270573e-08, 0.0, 0.0, 0.0, 2.0182994696828576e-15, 0.06977916421885891, 0.2579938169821856, 0.37671532887212333, 0.024565028057913885, 0.12617825469188657, 0.6340448467030686, 0.5754643941945634, 0.13048109721771595, 1.4341543078480776, 1.3216214397106167, 0.4415648326891368, 0.8216927039920507, 0.9032727935119587, 1.7656705299211204, 1.627512140396068, 1.5918752695978569, 2.051239519025683, 1.9657798505455604, 2.8356702084496526, 2.5590427999121284, 2.982526516926714, 2.6272105137933197, 2.960027241401188, 3.3362990605756897, 3.131536821850587, 3.159111523746814, 2.7752077904104517, 1.0508812833436538, 0.21463498939778325, 0.6073490769165661, 0.267278502966685, 0.3440305423365897, 1.0124744653679076, 1.342256457731422, 0.2717826360960419, 1.6632114845610049, 1.7120322558818795, 1.575579186781418, 2.202279058543633, 2.005531691208538, 1.891973046186819, 2.2982791457282916, 2.4252951011748562, 1.3467523990450576, 1.394123734140854, 1.306121994024973, 1.3563331357893305, 1.908587215779497, 2.2410142807813824, 2.1400334226110425, 2.0238641943935187, 2.4448713165408282, 0.001517101822563153, 9.482987339023494e-15, 0.0, 1.5827373963789938e-15, 1.580023867552888, 2.5839838573569534, 1.5871346370389885, 4.1337617125689725, 0.0016876790012300346, 2.337637442823331, 1.9268402331708496, 2.509223443618991, 2.8573979857307554, 2.7429627532040692, 2.3184117402885605, 2.2519888495692886, 1.441733890843454, 2.3283267069330638, 2.090507069768287, 1.616388780668859, 0.30852077577914405, 1.2418308849676503, 0.749579822648432, 2.0216862557918627, 1.8471265276557536, 1.9409844374088654, 2.029630658555306, 1.7488835640200255, 1.4429217698293368, 0.09853693097516952, 1.5685094105495399, 0.060035092997817674, 1.1562109869575399, 0.9883011451997243, 1.257630809337659, 1.6997562951967606, 0.4508041502784602, 0.3164090446061367, 1.4182969827012353, 1.3595162629204571, 1.475081471520821, 3.021289456736385, 3.0956206508951407, 2.481681959913101, 2.1308362149915583, 2.9008847410243757, 2.909122424144527, 2.7204695218309363, 2.087469496170989, 1.3538856364999683, 0.2008306196121235, 1.600964614957816, 1.4250387287265247, 1.6911311698607534, 1.1526705582269934, 0.7292975452608408, 0.4173852602091413, 0.2662677349829448, 1.5122097133171966, 1.0836674077065491, 1.031414686946613, 0.8173974802808452, 0.7095506482976092, 0.5949976898998941, 0.29670773819783247, 0.5083829941296425, 0.6440505445429058, 0.053833868680306, 7.794970013943188e-16, 0.7045208512938965, 1.6120699276991068, 1.5318231382310976, 1.7340273755678486, 2.4381603392169247, 2.6170276892155027, 2.5906147953962844, 3.407886846404895, 3.6212633825479905, 2.329416094716585, 2.958660792491976, 2.6669305434977773, 2.1590860132417564, 2.4937418622856145, 2.5589089762124786, 1.3338118137548678, 1.1942787402156965, 1.7418035300505763, 1.4081188229888932, 1.2487225960375548, 1.4492676314261193, 1.2654783371285574, 1.1685912162797853, 1.0148303874944786, 1.1962440020710763, 1.305708313372589, 0.6602148155530632, 0.337166044389861, 0.8396055147211476, 0.8562349502018103, 0.588778548048956, 0.7049070769530035, 1.2538977263308646, 1.4831897704424597, 1.4593441911500031, 2.1621717599542505, 2.4273857543891015, 2.426355640271325, 2.83832034285733, 2.7641303296460444, 2.2050969080359004, 2.6355562577584215, 3.1005626046243817, 2.4089187966341488, 1.8919645338161346, 1.8840157076492403, 1.344761629829863, 1.404294123950026, 1.8721961393692923, 1.3226408636613955, 0.4215497636181964, 0.5726863357586803, 1.0258923965461795, 1.1819610363504558, 0.8368490648663582, 0.6515561348082733, 0.6685731745760881, 0.5334870649826413, 0.8710519187832059, 0.6669646197224997, 0.5260752114304805, 0.3876797985565807, 0.03621327582895155, 0.46897871650384915, 0.8718533580569904, 0.7009452451531725, 1.4931853849244896, 1.8652719440498333, 1.6631794982365034, 1.494779190512575, 2.508688004725345, 3.0433643835464537, 3.2533878501144433, 3.579790260747532, 2.164640103097207, 0.6698924809914789, 2.1342050222506663, 2.5814605344559984, 1.6583152357630657, 1.3111552900920307, 1.20851491437197, 0.3334479204279151, 0.0027238985981172218, 0.7485037657977041, 0.23706880539492062, 0.3990097623354095, 4.751136081369487e-05, 1.5362095500430528, 0.46926869783190056, 0.0007246232360620678, 0.0, 5.239717734593537e-16, 9.938359637204445e-16, 0.0, 1.7385067095755083e-16, 4.106727240038999e-15, 0.10511094949367368, 0.026846967487429776, 0.0796163088839284, 0.8797518497354565, 2.616397453997683, 3.9912371044774604, 3.6233174077890604, 1.5672138389164023, 1.8304904251881515, 2.748948532497653, 3.287747311072218, 4.0926517829783675, 6.029811160308903];

  // Matolcsi Vinuesa 2010 solution
  const MATOLCSI_VINUESA_SOLUTION = [
    1.21174638, 0, 0, 0.25997048, 0.47606812,
    0.62295219, 0.3296586, 0, 0.29734381, 0,
    0, 0, 0, 0, 0,
    0, 0.00846453, 0.05731673, 0, 0.13014906,
    0, 0.08357863, 0.05268549, 0.06456956, 0.06158231,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 
    0.02396999, 0, 0, 0.05846552, 0,
    0, 0, 0, 0, 0.0026332,
    0.0509835, 0, 0.1283313, 0.0904924, 0.21232176,
    0.24866151, 0.09933512, 0.01963586, 0.01363895, 0.32389841,
    0, 0, 0.14467517, 0.0129752, 0,
    0, 0.16299837, 0.38329665, 0.11361262, 0.32074656,
    0.17344291, 0.33181372, 0.24357561, 0.2577003, 0.20567824,
    0.13085743, 0.17116496, 0.14349025, 0.07019695, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.0131741, 0.0342541, 0.0427565, 0.03045044,
    0.07900079, 0.07020678, 0.08528342, 0.09705597, 0.0932896,
    0.09360206, 0.06227754, 0.07943462, 0.08176106, 0.10667185,
    0.10178412, 0.11421821, 0.07773213, 0.11021377, 0.12190377,
    0.06572457, 0.07494855, 0, 0, 0.02140202,
    0, 0, 0.0231478, 0.00127997, 0,
    0.04672881, 0.03886266, 0.11141784, 0.00695668, 0.0466224,
    0.03543131, 0.08803511, 0.04165729, 0.10785652, 0.06747342,
    0.18785215, 0.31908323, 0.3249705, 0.09824861, 0.23309878,
    0.12428441, 0.03200975, 0.0933163, 0.09527521, 0.12202693,
    0.13179059, 0.09266878, 0.02013746, 0.16448047, 0.20324945,
    0.21810431, 0.27321179, 0.25242816, 0.19993811, 0.13683837,
    0.13304836, 0.08794214, 0.12893672, 0.16904485, 0.22510883,
    0.26079786, 0.27367504, 0.26271896, 0.20457964, 0.15073917,
    0.11014028, 0.09896, 0.0926069, 0.13269111, 0.17329988,
    0.20761774, 0.21707182, 0.18933169, 0.14601258, 0.08531506,
    0.06187865, 0.06100211, 0.09064962, 0.12781018, 0.17038096,
    0.185766, 0.1734501, 0.14667009, 0.09569536, 0.06092822,
    0.03219067, 0.0495587, 0.09657756, 0.16382398, 0.22606693,
    0.22230709, 0.19833621, 0.16155032, 0.09330751, 0.02838363,
    0.02769322, 0.03349924, 0.09448887, 0.20517242, 0.22849741,
    0.24175836, 0.19700135, 0.18168723
  ];

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
  
  // Move these up before the functions are defined
  const [isInitialized, setIsInitialized] = useState(false);
  const [lastManualPiecesChange, setLastManualPiecesChange] = useState(0);
  
  // We'll add the initialization effect after all the functions are defined
  
  // Update max autoconvolution value when autoconvolution changes
  useEffect(() => {
    if (autoconvolution.length > 0) {
      const maxVal = Math.max(...autoconvolution.map(point => point.y));
      setMaxAutoconvValue(maxVal);
    }
  }, [autoconvolution]);
  
  // Define calculation functions separately to avoid circular dependencies
  
  // Calculate autoconvolution without dependencies on React state
  const calculateAutoconvolutionData = (steps) => {
    // Use the actual number of steps
    const P = steps.length;
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
    
    return result;
  };
  
  // Calculate total height
  const calculateTotalHeight = (steps) => {
    return steps.reduce((acc, step) => acc + step.y, 0);
  };
  
  // Wrapper function that updates state
  const calculateAutoconvolution = useCallback((steps) => {
    const result = calculateAutoconvolutionData(steps);
    setAutoconvolution(result);
  }, []);
  
  // Update total height
  const updateTotalHeight = useCallback((steps) => {
    const sum = calculateTotalHeight(steps);
    setTotalHeight(sum);
  }, []);
  
  // Define function to create a step function from heights
  const createStepFunction = (heights) => {
    const pieces = heights.length;
    const pieceWidth = (MAX_X - MIN_X) / pieces;
    let newStepFunction = [];
    
    for (let i = 0; i < pieces; i++) {
      const x = MIN_X + (i + 0.5) * pieceWidth;
      newStepFunction.push({
        x,
        y: heights[i], 
        pieceIndex: i,
        width: pieceWidth
      });
    }
    
    return newStepFunction;
  };

  // Function to load predefined solutions
  const loadPredefinedSolution = useCallback((type) => {
    const heights = type === 'google' ? GOOGLE_SOLUTION : MATOLCSI_VINUESA_SOLUTION;
    const newStepFunction = createStepFunction(heights);
    
    // Update everything at once
    setStepFunction(newStepFunction);
    setSelectedPiece(null);
    setCurrentHeight(0);
    setNumPieces(heights.length);
    
    // Calculate derived values
    const result = calculateAutoconvolutionData(newStepFunction);
    setAutoconvolution(result);
    setTotalHeight(calculateTotalHeight(newStepFunction));
  }, []);
  
  // Generate random step function
  const generateRandomStepFunction = useCallback(() => {
    const pieces = numPieces; // Use current number of pieces
    const pieceWidth = (MAX_X - MIN_X) / pieces;
    let heights = [];
    
    // Generate random heights
    for (let i = 0; i < pieces; i++) {
      heights.push(Math.random() * (MAX_HEIGHT / 5)); // Using MAX_HEIGHT/5 for initial values to leave room for adjustment
    }
    
    // Create step function
    const newStepFunction = createStepFunction(heights);
    
    // Update state
    setStepFunction(newStepFunction);
    setSelectedPiece(null);
    setCurrentHeight(0);
    
    // Calculate derived values
    const result = calculateAutoconvolutionData(newStepFunction);
    setAutoconvolution(result);
    setTotalHeight(calculateTotalHeight(newStepFunction));
  }, [numPieces]);
  
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
      const result = calculateAutoconvolutionData(updated);
      setAutoconvolution(result);
      setTotalHeight(calculateTotalHeight(updated));
      
      return updated;
    });
  }, [selectedPiece]);
  
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
        const result = calculateAutoconvolutionData(updated);
        setAutoconvolution(result);
        setTotalHeight(calculateTotalHeight(updated));
        
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
  }, [selectedPiece, currentHeight, handlePieceClick]);
  
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
                      data-index={index}
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
  
  // Add the initialization effect now that all functions are defined
  useEffect(() => {
    if (!isInitialized) {
      setIsInitialized(true);
      generateRandomStepFunction();
    }
  }, [isInitialized, generateRandomStepFunction]);
  
  // Effect to update when slider changes
  useEffect(() => {
    if (lastManualPiecesChange > 0) {
      generateRandomStepFunction();
    }
  }, [lastManualPiecesChange, generateRandomStepFunction]);

  return (
    <div className="flex flex-col p-4 space-y-4 max-w-5xl mx-auto">
      <div className="text-2xl font-bold text-center">Step Function Autoconvolution</div>
      
      {/* Controls */}
      <div className="bg-white p-4 rounded-lg shadow">
        <div className="grid grid-cols-2 gap-6">
          {/* Left column - Sliders */}
          <div className="flex flex-col space-y-6">
            <div>
              <div className="mb-2 font-medium">Number of Pieces: {numPieces}</div>
              <div className="relative w-full">
                <input
                  type="range"
                  min="5"
                  max={MAX_PIECES}
                  value={numPieces}
                  onChange={(e) => {
                    // Update the number of pieces
                    setNumPieces(Number(e.target.value));
                    // Track that this was a manual change
                    setLastManualPiecesChange(Date.now());
                  }}
                  className="w-full slider-thumb-orange"
                  style={{
                    background: `linear-gradient(to right, #8884d8 0%, #8884d8 ${(numPieces / MAX_PIECES) * 100}%, #e5e7eb ${(numPieces / MAX_PIECES) * 100}%, #e5e7eb 100%)`
                  }}
                />
              </div>
            </div>
            
            <div>
              <div className="mb-2 font-medium">Height Adjustment: {currentHeight.toFixed(2)}</div>
              <div className="relative w-full">
                <input
                  type="range"
                  min="0"
                  max={MAX_HEIGHT * 5}
                  step="0.1"
                  value={currentHeight}
                  disabled={selectedPiece === null}
                  className="w-full slider-thumb-orange"
                  onInput={(e) => {
                    if (selectedPiece === null) return;
                    
                    // Get new height value from slider
                    const newHeight = Number(e.target.value);
                    
                    // Update current height value
                    setCurrentHeight(newHeight);
                    
                    // Update both step function and display in sync
                    setStepFunction(prev => {
                      const updated = [...prev];
                      updated[selectedPiece] = {
                        ...updated[selectedPiece],
                        y: newHeight
                      };
                      
                      // Calculate autoconvolution in the same update
                      // This ensures the step function and autoconvolution stay in sync
                      calculateAutoconvolution(updated);
                      updateTotalHeight(updated);
                      
                      return updated;
                    });
                  }}
                  style={{
                    background: selectedPiece !== null ? 
                      `linear-gradient(to right, #ff7300 0%, #ff7300 ${(currentHeight / (MAX_HEIGHT * 5)) * 100}%, #e5e7eb ${(currentHeight / (MAX_HEIGHT * 5)) * 100}%, #e5e7eb 100%)` : 
                      '#e5e7eb'
                  }}
                />
              </div>
            </div>
          </div>
          
          {/* Right column - Buttons and Info */}
          <div className="flex flex-col justify-between">
            <div className="flex flex-col space-y-2">
              <div className="font-medium mb-2">Selected Piece: {selectedPiece !== null ? selectedPiece : 'None'}</div>
              <div className="flex flex-col space-y-2">
                <div className="flex space-x-2">
                  <button
                    onClick={generateRandomStepFunction}
                    className="bg-blue-500 text-white px-3 py-2 rounded hover:bg-blue-600"
                  >
                    Random
                  </button>
                  
                  <button
                    onClick={() => loadPredefinedSolution('google')}
                    className="bg-green-600 text-white px-3 py-2 rounded hover:bg-green-700"
                  >
                    Google (600)
                  </button>
                  
                  <button
                    onClick={() => loadPredefinedSolution('matolcsi')}
                    className="bg-purple-600 text-white px-3 py-2 rounded hover:bg-purple-700"
                  >
                    Matolcsi-Vinuesa (208)
                  </button>
                </div>
                
                {selectedPiece !== null && (
                  <button
                    onClick={handlePieceDeselect}
                    className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 w-full"
                  >
                    Deselect
                  </button>
                )}
              </div>
            </div>
            
            <div className="text-sm text-gray-700 mt-4">
              <div className="font-semibold">Instructions:</div>
              <ol className="list-decimal pl-5 mt-1">
                <li>Click on any bar to select it (turns orange)</li>
                <li>Drag selected bar up/down to adjust height or use the slider</li>
                <li>The autoconvolution plot shows Google's bound at 1.5053</li>
              </ol>
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
      
    </div>
  );
};

export default StepFunctionVisualizer;