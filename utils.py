import torch
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import bfgs_weak_wolfe
import ntd
import cvxpy as cp


def return_height_params_google_simplex():
    height_params_google = torch.tensor([9.00854017233134, 4.581788824134047, 5.954983797866223, 3.7314786029786324, 4.25817159660483, 3.544987049547799, 0.08876194700959494, 0.0491697316439318, 1.7439263266999894, 3.8548164683263795, 3.621038728073569, 4.04587668218835, 4.68211946529981, 5.5904236142896675, 4.737832747546433, 3.1594093823451055, 1.902874129984629, 2.7870307391136304, 3.277574995692391, 1.8981329099596054, 1.526040859367755, 2.305128838504833, 5.17673786436095, 4.583218228762042, 3.9910761392791887, 2.784600928752006, 5.450687602543662, 6.170368723277989, 7.045569321986071e-16, 7.149948549556939e-15, 0.0, 0.0, 0.0, 0.0, 1.2580295353835013e-15, 0.0, 0.0, 0.0, 0.0, 3.873037303627252e-15, 0.0, 0.0, 2.020385656008168e-06, 0.000293922119342568, 0.0, 4.9514916125368726e-15, 7.282654612521097e-16, 1.906059354629418e-14, 0.0, 3.3528418595404916e-15, 1.5099558045700925e-15, 4.901439953827422e-15, 0.0, 8.851999542886555e-15, 0.0, 0.0, 0.0005211322699854395, 0.3757576289315001, 0.25176470069965495, 4.1179587840945515e-06, 0.0, 2.946431316197597e-15, 0.0, 1.0333089131925899e-16, 2.591940622467849e-15, 0.0, 6.852171628124262e-15, 0.0, 0.0, 1.3885601200927435e-14, 2.5015636739088256e-15, 1.4382184696274247e-14, 1.235388698636516e-15, 9.328196456283097e-15, 6.938490364750181e-15, 5.581796597296351e-17, 0.0, 0.0, 5.1220388613389905e-15, 0.0, 6.085199919293191e-15, 0.0, 0.0, 1.0633201915504476e-14, 6.240893078396387e-16, 0.0, 9.242385301100576e-15, 2.1818685641605435e-15, 0.0, 3.841626602268906e-15, 0.0013592097228050644, 8.120066974555713e-15, 8.479388423870961e-16, 2.5924005380166956e-15, 0.0, 2.6610672065525727e-15, 0.0, 1.233819156251431e-14, 8.819083406210366e-15, 0.0, 4.492323424835768e-16, 0.0, 3.0916450306058138e-15, 0.0, 0.0, 3.404186949211756e-15, 4.54126650881379e-15, 1.462631558763152e-14, 0.0, 0.0, 0.0, 1.4460597710909072e-15, 9.521734973996671e-15, 0.0, 4.559858799705722e-15, 7.864867909828807e-16, 0.0, 1.7856864350178655e-16, 0.00021045010164189585, 0.26541232693216404, 0.8094426381528257, 0.5750041584597478, 0.23313281323505236, 3.6007277514467585e-05, 0.0, 0.7828826491881691, 0.43382874037802, 1.3263698571911402, 0.5441713262465393, 0.9864380574571914, 0.6776516652004773, 0.5910950602641856, 0.507419190418916, 0.5231329501406576, 0.9391246115133585, 0.4508771959372286, 0.28283039994676146, 1.2889986480406397, 0.9649046182943108, 1.4104382244415803, 1.3916682358533747, 0.8743196646011149, 0.7627485335443527, 0.2103862254578538, 0.14545209168646947, 0.019762475547189184, 1.2279396984729254, 0.012006361768949678, 1.7677675926679783, 0.9303739918691369, 1.0966313889580412, 0.40142701455261154, 0.1477985748190306, 0.1310850821272394, 0.0027642064206369592, 0.6718883532064702, 0.287789791442545, 1.1886491680958895, 0.6459736548490735, 0.88966666001013, 0.36931312374260505, 0.6840914190936884, 0.38692129734520775, 0.8050006872194091, 0.26610729268169875, 0.002941709304056364, 0.5150673486621109, 0.4049854152265144, 1.1607178193685956, 1.7547854228356075, 0.0, 0.8531817250969695, 2.3845552035650363e-05, 0.035208188035124974, 0.06799207369201249, 0.14050016250524128, 0.4862562534194792, 1.508781726996261, 0.46943710673489225, 0.22962993226722195, 1.589825945710927e-11, 3.51517770993058e-15, 2.4398590319680178e-15, 1.1666504235544564e-06, 0.0021946672216711, 0.34171503722540436, 0.4703022197366691, 0.1313974666218601, 0.11754826815054241, 0.0, 2.2387234387833643e-16, 0.0, 7.192783695625604e-05, 0.4486935802226264, 1.234691190028419, 2.8985055264499153, 1.0234017394012231, 2.7375379465420373, 0.5899927642043619, 1.4461499611411766, 0.7033498408537826, 1.6505029216035125, 0.9593634797752735, 0.009302210703764222, 0.0004181359389419785, 0.0, 0.0, 0.0023430720926212976, 0.42801036705183393, 0.6031743194865573, 1.8862845950884395, 1.0944504439060767, 1.3978223736063145, 0.13603422891356853, 0.8568768273359568, 0.5287328963079988, 0.04201038853661816, 0.5746932650501643, 0.7698787794362285, 2.2478052766496255, 1.3267115762262056, 1.3819155415467284, 1.210307904386098, 1.2050374056121944, 0.973960636675429, 0.13506178694552, 0.0017211602091930576, 1.2080793667302383, 0.9431703684918005, 0.004927152124127672, 0.26457949335968395, 0.219096730428291, 0.8972094379125464, 1.009247390062118, 2.5396761105116816, 2.0567929964131704, 2.5384945885180765, 2.051772820060434, 2.841483226472209, 2.5484575236736253, 2.900405077014117, 2.7293223781158513, 2.8016507480694623, 2.5235338506952227, 2.842495616436774, 3.6113040879253218, 2.4409992918997654, 2.8613737519007785, 2.0376653653073236, 2.873716631081072, 2.7431139992026585, 2.3176851657187343, 2.963845077577065, 2.1297112056154828, 3.1281786712157276, 1.559962066888169, 1.5175735153572592, 1.8986372289826554, 2.422172211485286, 1.4024751115172904, 1.6645681102200025, 1.0890488631004245, 0.9551468779062758, 0.4210663124027455, 0.7844656815643463, 1.3849725648239561, 1.1400002207678432, 1.2589535564861496, 0.00010847583255872839, 0.33022246693439483, 0.009991411612394792, 3.897603693807049e-14, 0.0, 0.0, 4.615098985648224e-16, 0.0, 0.0, 0.00019552451607645426, 7.535959259635103e-15, 0.0, 0.0, 0.0, 3.391322478636254e-15, 0.0, 5.23262076392476e-16, 0.0, 0.0013138779575907105, 3.8754445450268335e-05, 0.6732744675805906, 0.1403000505612687, 0.1066587972153134, 3.248799681911591e-05, 0.0002955183459383166, 0.003362359949825627, 1.258993392593319, 0.6721840514813638, 1.4023077519116312, 0.26971646568749497, 1.5317811712427716e-06, 0.0, 0.0004643255998670316, 0.21977013225378167, 0.4192480303816599, 1.5193826730071023, 1.140986767032434e-14, 0.00023172541086911792, 0.0, 2.8126101180648514e-16, 0.3250641832764737, 0.2631639184319745, 9.581064115586714e-06, 6.911241239536039e-08, 2.1977413193942923e-15, 0.0, 0.009284626840990289, 0.38181362260215157, 0.9229241174303233, 1.2389265199825776, 4.003701357117134e-15, 0.0018347819223694948, 0.09741977675536181, 0.7177732728524777, 0.893918190086622, 0.7592094697415344, 0.33941833076021116, 0.9895237780462212, 1.2103201657651075, 0.1562357060123791, 0.05779942219135546, 0.046035988201598356, 0.20605598834789451, 0.7087789468745198, 0.9835248618606901, 0.15343068950958472, 0.7458575667928422, 1.1650897270704061, 0.8603305066335794, 0.21035575411632462, 0.1359607526732069, 0.27083984880209855, 0.6401471037748572, 3.021195033008772e-11, 0.2302514662496976, 0.6133656015904811, 0.0014526177283533382, 0.2833008515927792, 0.6173666814379335, 0.35174441102605253, 0.42994053621927014, 0.17019592622467353, 0.18730260272562482, 0.20732692125660646, 0.0006488239072035948, 0.0008689690193708769, 0.0060827028748053225, 0.42803050608386095, 0.5802023431307022, 0.07735146147149399, 0.2677857202504946, 0.0009184606747345728, 0.43037124020293993, 0.3617534843366182, 0.3772422594534307, 0.2584947755294139, 0.001169636807952639, 1.9643151751270573e-08, 0.0, 0.0, 0.0, 2.0182994696828576e-15, 0.06977916421885891, 0.2579938169821856, 0.37671532887212333, 0.024565028057913885, 0.12617825469188657, 0.6340448467030686, 0.5754643941945634, 0.13048109721771595, 1.4341543078480776, 1.3216214397106167, 0.4415648326891368, 0.8216927039920507, 0.9032727935119587, 1.7656705299211204, 1.627512140396068, 1.5918752695978569, 2.051239519025683, 1.9657798505455604, 2.8356702084496526, 2.5590427999121284, 2.982526516926714, 2.6272105137933197, 2.960027241401188, 3.3362990605756897, 3.131536821850587, 3.159111523746814, 2.7752077904104517, 1.0508812833436538, 0.21463498939778325, 0.6073490769165661, 0.267278502966685, 0.3440305423365897, 1.0124744653679076, 1.342256457731422, 0.2717826360960419, 1.6632114845610049, 1.7120322558818795, 1.575579186781418, 2.202279058543633, 2.005531691208538, 1.891973046186819, 2.2982791457282916, 2.4252951011748562, 1.3467523990450576, 1.394123734140854, 1.306121994024973, 1.3563331357893305, 1.908587215779497, 2.2410142807813824, 2.1400334226110425, 2.0238641943935187, 2.4448713165408282, 0.001517101822563153, 9.482987339023494e-15, 0.0, 1.5827373963789938e-15, 1.580023867552888, 2.5839838573569534, 1.5871346370389885, 4.1337617125689725, 0.0016876790012300346, 2.337637442823331, 1.9268402331708496, 2.509223443618991, 2.8573979857307554, 2.7429627532040692, 2.3184117402885605, 2.2519888495692886, 1.441733890843454, 2.3283267069330638, 2.090507069768287, 1.616388780668859, 0.30852077577914405, 1.2418308849676503, 0.749579822648432, 2.0216862557918627, 1.8471265276557536, 1.9409844374088654, 2.029630658555306, 1.7488835640200255, 1.4429217698293368, 0.09853693097516952, 1.5685094105495399, 0.060035092997817674, 1.1562109869575399, 0.9883011451997243, 1.257630809337659, 1.6997562951967606, 0.4508041502784602, 0.3164090446061367, 1.4182969827012353, 1.3595162629204571, 1.475081471520821, 3.021289456736385, 3.0956206508951407, 2.481681959913101, 2.1308362149915583, 2.9008847410243757, 2.909122424144527, 2.7204695218309363, 2.087469496170989, 1.3538856364999683, 0.2008306196121235, 1.600964614957816, 1.4250387287265247, 1.6911311698607534, 1.1526705582269934, 0.7292975452608408, 0.4173852602091413, 0.2662677349829448, 1.5122097133171966, 1.0836674077065491, 1.031414686946613, 0.8173974802808452, 0.7095506482976092, 0.5949976898998941, 0.29670773819783247, 0.5083829941296425, 0.6440505445429058, 0.053833868680306, 7.794970013943188e-16, 0.7045208512938965, 1.6120699276991068, 1.5318231382310976, 1.7340273755678486, 2.4381603392169247, 2.6170276892155027, 2.5906147953962844, 3.407886846404895, 3.6212633825479905, 2.329416094716585, 2.958660792491976, 2.6669305434977773, 2.1590860132417564, 2.4937418622856145, 2.5589089762124786, 1.3338118137548678, 1.1942787402156965, 1.7418035300505763, 1.4081188229888932, 1.2487225960375548, 1.4492676314261193, 1.2654783371285574, 1.1685912162797853, 1.0148303874944786, 1.1962440020710763, 1.305708313372589, 0.6602148155530632, 0.337166044389861, 0.8396055147211476, 0.8562349502018103, 0.588778548048956, 0.7049070769530035, 1.2538977263308646, 1.4831897704424597, 1.4593441911500031, 2.1621717599542505, 2.4273857543891015, 2.426355640271325, 2.83832034285733, 2.7641303296460444, 2.2050969080359004, 2.6355562577584215, 3.1005626046243817, 2.4089187966341488, 1.8919645338161346, 1.8840157076492403, 1.344761629829863, 1.404294123950026, 1.8721961393692923, 1.3226408636613955, 0.4215497636181964, 0.5726863357586803, 1.0258923965461795, 1.1819610363504558, 0.8368490648663582, 0.6515561348082733, 0.6685731745760881, 0.5334870649826413, 0.8710519187832059, 0.6669646197224997, 0.5260752114304805, 0.3876797985565807, 0.03621327582895155, 0.46897871650384915, 0.8718533580569904, 0.7009452451531725, 1.4931853849244896, 1.8652719440498333, 1.6631794982365034, 1.494779190512575, 2.508688004725345, 3.0433643835464537, 3.2533878501144433, 3.579790260747532, 2.164640103097207, 0.6698924809914789, 2.1342050222506663, 2.5814605344559984, 1.6583152357630657, 1.3111552900920307, 1.20851491437197, 0.3334479204279151, 0.0027238985981172218, 0.7485037657977041, 0.23706880539492062, 0.3990097623354095, 4.751136081369487e-05, 1.5362095500430528, 0.46926869783190056, 0.0007246232360620678, 0.0, 5.239717734593537e-16, 9.938359637204445e-16, 0.0, 1.7385067095755083e-16, 4.106727240038999e-15, 0.10511094949367368, 0.026846967487429776, 0.0796163088839284, 0.8797518497354565, 2.616397453997683, 3.9912371044774604, 3.6233174077890604, 1.5672138389164023, 1.8304904251881515, 2.748948532497653, 3.287747311072218, 4.0926517829783675, 6.029811160308903])

    height_params_google = (1200.0)*height_params_google / (torch.sum(height_params_google))
    height_params_google.requires_grad = True
    return height_params_google

def initialize_from_matolcsi_vinuesa_analytic(P_val: int) -> torch.Tensor:
    """
    Initializes step function heights h based on the Matolcsi & Vinuesa 2010 analytic function.
    The function is sampled at the midpoint of each of the P_val intervals.
    The resulting heights are then scaled to sum to S_target.
    """
    # Define the x-coordinates for the midpoints of each interval
    # Interval is [-0.25, 0.25]
    # delta_x = 0.5 / P_val
    # x_coords_mid = -0.25 + delta_x/2 + torch.arange(P_val) * delta_x
    S_target = 2*P_val
    
    # More robust way to get midpoints:
    step_edges = torch.linspace(-0.25, 0.25, P_val + 1, dtype=torch.float64)
    x_coords_mid = (step_edges[:-1] + step_edges[1:]) / 2.0

    heights_analytic = torch.zeros(P_val, dtype=torch.float64)

    for i, x_mid in enumerate(x_coords_mid):
        if -0.25 < x_mid < 0:
            # Ensure argument to power is non-negative; small epsilon if x_mid is very close to -0.25
            base = torch.clamp(0.500166 + 2 * x_mid, min=1e-12) 
            heights_analytic[i] = 0.338537 / (base**0.65)
        elif 0 < x_mid < 0.25:
             # Ensure argument to power is non-negative; small epsilon if x_mid is very close to 0
            base = torch.clamp(0.00195 + 2 * x_mid, min=1e-12)
            heights_analytic[i] = 1.392887 / (base**(1/3))
        elif x_mid == 0:
            # The function is not strictly defined at x=0.
            # We can average the limits, or take one side, or set to a reasonable value.
            # For simplicity, let's use the limit from the right (as it's larger).
            # Or, if P_val is even, x_mid will never be exactly 0.
            # If P_val is odd, one x_mid could be 0.
            # Taking limit from right:
            if P_val % 2 != 0 and x_mid == 0: # Only if an interval midpoint is exactly 0
                base = torch.clamp(0.00195 + 2 * (x_mid + 1e-9), min=1e-12) # slightly to the right of 0
                heights_analytic[i] = 1.392887 / (base**(1/3))
            else: # Should not happen for x_mid == 0 if P is even.
                  # Default to 0 if x_mid is exactly -0.25 or 0.25, or 0 if P is even.
                heights_analytic[i] = 0.0
        else: # x_mid is exactly -0.25 or 0.25 or outside (-0.25, 0.25) - should not happen for midpoints
            heights_analytic[i] = 0.0
            
    # Ensure non-negativity due to potential floating point issues near boundaries or clamp
    heights_analytic = torch.clamp(heights_analytic, min=0.0)

    # Scale so that sum(heights_analytic) = S_target
    current_sum = torch.sum(heights_analytic)
    if current_sum > 1e-9: # Avoid division by zero if all sampled heights are zero
        heights_normalized = (S_target / current_sum) * heights_analytic
    else:
        # Fallback: if function samples to near zero everywhere (e.g. P_val is tiny or issue)
        # initialize to a uniform distribution.
        print("Warning: Analytic function sampled to near zero sum. Initializing uniformly.")
        heights_normalized = torch.full((P_val,), S_target / P_val, dtype=torch.float64)
        
    return heights_normalized


def initialize_from_matolcsi_vinuesa_best_step_208(S_target: float) -> torch.Tensor:
    """
    Initializes step function heights h based on the best non-negative step function
    reported by Matolcsi & Vinuesa (2010) with n=208 pieces.
    The P_val for this function is fixed at 208.
    The reported coefficients are scaled to sum to S_target.
    """
    P_val = 208 # This is fixed by the paper's data

    # Corrected and complete coefficients from the paper (Appendix A, n=208)
    # Data provided by user.
    coeffs_str = """
    1.21174638 0. 0. 0.25997048 0.47606812
    0.62295219 0.3296586 0. 0.29734381 0.
    0. 0. 0. 0. 0.
    0. 0.00846453 0.05731673 0. 0.13014906
    0. 0.08357863 0.05268549 0.06456956 0.06158231
    0. 0. 0. 0. 0.
    0. 0. 0. 0. 0.
    0. 0. 0. 0. 0.
    0. 0. 0. 0. 0. 
    0.02396999 0. 0. 0.05846552 0.
    0. 0. 0. 0. 0.0026332
    0.0509835 0. 0.1283313 0.0904924 0.21232176
    0.24866151 0.09933512 0.01963586 0.01363895 0.32389841
    0. 0. 0.14467517 0.0129752 0.
    0. 0.16299837 0.38329665 0.11361262 0.32074656
    0.17344291 0.33181372 0.24357561 0.2577003 0.20567824
    0.13085743 0.17116496 0.14349025 0.07019695 0.
    0. 0. 0. 0. 0.
    0. 0. 0. 0. 0.
    0. 0. 0. 0. 0.
    0. 0.0131741 0.0342541 0.0427565 0.03045044
    0.07900079 0.07020678 0.08528342 0.09705597 0.0932896
    0.09360206 0.06227754 0.07943462 0.08176106 0.10667185
    0.10178412 0.11421821 0.07773213 0.11021377 0.12190377
    0.06572457 0.07494855 0. 0. 0.02140202
    0. 0. 0.0231478 0.00127997 0.
    0.04672881 0.03886266 0.11141784 0.00695668 0.0466224
    0.03543131 0.08803511 0.04165729 0.10785652 0.06747342
    0.18785215 0.31908323 0.3249705 0.09824861 0.23309878
    0.12428441 0.03200975 0.0933163 0.09527521 0.12202693
    0.13179059 0.09266878 0.02013746 0.16448047 0.20324945
    0.21810431 0.27321179 0.25242816 0.19993811 0.13683837
    0.13304836 0.08794214 0.12893672 0.16904485 0.22510883
    0.26079786 0.27367504 0.26271896 0.20457964 0.15073917
    0.11014028 0.09896 0.0926069 0.13269111 0.17329988
    0.20761774 0.21707182 0.18933169 0.14601258 0.08531506
    0.06187865 0.06100211 0.09064962 0.12781018 0.17038096
    0.185766 0.1734501 0.14667009 0.09569536 0.06092822
    0.03219067 0.0495587 0.09657756 0.16382398 0.22606693
    0.22230709 0.19833621 0.16155032 0.09330751 0.02838363
    0.02769322 0.03349924 0.09448887 0.20517242 0.22849741
    0.24175836 0.19700135 0.18168723
    """ # The 42nd line has 5 elements, the 43rd (last data) line has 3.
        # Total lines of numbers: 41 full lines of 5 numbers = 205.
        # Plus the last line of 3 numbers = 208.

    raw_coeffs = []
    for line in coeffs_str.strip().split('\n'):
        # Skip empty lines that might result from an extra newline in the string
        if not line.strip():
            continue
        try:
            raw_coeffs.extend([float(x) for x in line.split()])
        except ValueError as e:
            print(f"Error parsing line: '{line}'")
            print(f"ValueError: {e}")
            raise
    
    if len(raw_coeffs) != P_val:
        # This check is crucial. If it fails, the copy-paste or parsing is still off.
        print(f"CRITICAL PARSING ERROR: Expected {P_val} coefficients for P_val={P_val}, but parsed {len(raw_coeffs)}.")
        print("Please meticulously re-check the coeffs_str data against the paper's Appendix A for n=208.")
        print("Parsed data (first 10):", raw_coeffs[:10])
        print("Parsed data (last 10):", raw_coeffs[-10:])
        # Fallback to uniform to prevent crashes, but this indicates a data entry problem.
        heights_paper = torch.full((P_val,), S_target / P_val, dtype=torch.float64)
        # Or raise ValueError to stop execution until data is fixed
        # raise ValueError(f"Coefficient parsing error: Expected {P_val}, got {len(raw_coeffs)}")
    else:
        heights_paper = torch.tensor(raw_coeffs, dtype=torch.float64)

    current_sum_paper = torch.sum(heights_paper)
    
    if current_sum_paper.abs().item() > 1e-9: # Use abs() for sum check
        heights_normalized = (S_target / current_sum_paper) * heights_paper
    else:
        # This case should be less likely if the paper's coefficients are generally positive
        # and the parsing is correct.
        print(f"Warning: Paper coefficients (P_val={P_val}) summed to near zero ({current_sum_paper.item()}). Initializing uniformly.")
        heights_normalized = torch.full((P_val,), S_target / P_val, dtype=torch.float64)
        
    return torch.clamp(heights_normalized, min=0.0) # Ensure non-negativity

### Step function opts ######

# Step function representation:
# The step function f is implicitly defined by:
# - heights: a 1D tensor of P non-negative values h_i.
# - P: number of pieces.
# - interval_range: a tuple (x_min, x_max), e.g., (-0.25, 0.25).
# - delta_x: width of each piece, (x_max - x_min) / P.
# - x_coords: tensor of starting x-coordinates for each piece.
# For optimization, 'heights' would be your learnable parameters.
#
# Example of how you might define these:
# P = 600
# interval_range = (-0.25, 0.25)
# heights = torch.rand(P, dtype=torch.float64, requires_grad=True) # example heights
# delta_x = (interval_range[1] - interval_range[0]) / P
# x_min = interval_range[0]

def compute_integral_of_step_function(heights: torch.Tensor, delta_x: float) -> torch.Tensor:
    """
    Computes the integral of the step function f(x)dx.
    f(x) is defined by `heights` h_i over pieces of width `delta_x`.
    Integral = delta_x * sum(h_i).
    """
    if not torch.all(heights >= 0):
        # print("warning: heights should be non-negative for the problem context.")
        # depending on your optimization strategy, you might enforce this elsewhere (e.g. h_i = relu(alpha_i))
        pass # up to you how strict to be here, problem says "non-negative f"
    return delta_x * torch.sum(heights)

def compute_autoconvolution_values(heights: torch.Tensor, delta_x: float, P: int) -> torch.Tensor:
    """
    Computes the values of the autoconvolution (f*f)(t) at the knot points.
    (f*f)(t) is piecewise linear. Max value occurs at one of these knots.
    Knots are t_m = 2*x_min + m*delta_x for m = 0, ..., 2P.
    Values are [0, delta_x * (H*H)_0, ..., delta_x * (H*H)_{2P-2}, 0].
    (H*H) is the discrete convolution of the height sequence H.
    """
    # Ensure heights is 1D
    if heights.ndim != 1 or heights.shape[0] != P:
        raise ValueError(f"heights tensor must be 1D with length P={P}. Got shape {heights.shape}")

    # Reshape heights for conv1d: (batch_size, C_in, L_in)
    # batch_size=1, C_in=1
    h_signal = heights.view(1, 1, P)
    
    # The kernel for conv1d to compute (H*H) should be H flipped.
    # weight for conv1d: (C_out, C_in/groups, L_kernel)
    # C_out=1, C_in/groups=1
    h_kernel_flipped = torch.flip(heights, dims=[0]).view(1, 1, P)

    # Compute H*H using conv1d. Padding P-1 results in output length 2P-1.
    # These are (H*H)_0, ..., (H*H)_{2P-2}
    conv_result = F.conv1d(h_signal, h_kernel_flipped, padding=P-1).squeeze()
    
    # Scale by delta_x
    conv_scaled = delta_x * conv_result
    
    # Add zeros for (f*f)(t_0) and (f*f)(t_{2P})
    zero = torch.tensor([0.0], device=heights.device, dtype=heights.dtype)
    autoconvolution_knot_values = torch.cat([zero, conv_scaled, zero])
    
    return autoconvolution_knot_values

### End Integral of step function end ###


### Plotting ###



# --- plotting functions ---
def plot_rendered_step_function(heights_numpy: np.ndarray, interval: tuple[float, float], title=""):
    """plots a step function f(x) cleanly using plt.step."""
    P = len(heights_numpy)
    x_min, x_max = interval
    
    step_edges = np.linspace(x_min, x_max, P + 1, dtype=float) # ensure float for plotting
    
    plt.figure(figsize=(8, 5))
    plt.step(step_edges[:-1], heights_numpy, where='post', color='blue', linewidth=1.5)
    
    plt.axhline(0, color='black', linewidth=0.5) # reference line at y=0
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    
    # dynamic axis limits
    x_padding = 0.05 * (x_max - x_min) if (x_max - x_min) > 0 else 0.05
    plt.xlim([x_min - x_padding, x_max + x_padding])
    
    if P > 0:
        max_h = np.max(heights_numpy)
        min_h = np.min(heights_numpy) # relevant if heights could be negative
        if max_h > 0: # typical case for non-negative heights
            plt.ylim([-0.1 * max_h, max_h * 1.2])
        elif max_h == 0 and min_h == 0 : # all zero heights
            plt.ylim([-0.5, 0.5])
        else: # other cases (e.g. all negative, not expected here)
            plt.ylim([min_h - 0.1*abs(min_h), max_h + 0.1*abs(max_h)])

    else: # P=0, no data
        plt.ylim([-0.5, 1.0]) 
        
    plt.grid(True)
    plt.show()

def plot_rendered_convolution(t_values_numpy: np.ndarray, conv_values_numpy: np.ndarray, title=""):
    """plots a piecewise linear function, e.g., the autoconvolution f*f(t)."""
    plt.figure(figsize=(8, 5))
    plt.plot(t_values_numpy, conv_values_numpy, marker='o', linestyle='-', color='green', markersize=3, linewidth=1.5)
    
    plt.xlabel("t")
    plt.ylabel("f*f(t)")
    plt.title(title)
    
    if len(t_values_numpy) > 0:
        t_min, t_max = t_values_numpy[0], t_values_numpy[-1]
        t_padding = 0.05 * (t_max - t_min) if (t_max - t_min) > 0 else 0.05
        plt.xlim([t_min - t_padding, t_max + t_padding])
        
        max_conv = np.max(conv_values_numpy)
        min_conv = np.min(conv_values_numpy)
        # autoconvolution of non-negative f(x) is non-negative
        if max_conv > 0:
            plt.ylim([-0.1 * max_conv, max_conv * 1.2])
        else: # all zero (or P=0 for f(x))
            plt.ylim([-0.5, 0.5])
    else: # no data
        plt.xlim([-0.55, 0.55]) # default based on expected autoconv range
        plt.ylim([-0.1, 1.0])
        
    plt.grid(True)
    plt.show()

###   

### Create loss functions

def constrained_loss(P_val=600):
    f_interval = (-0.25, 0.25) # interval for f(x)
    f_x_min, f_x_max = f_interval
    f_delta_x = (f_x_max - f_x_min) / P_val
    return lambda h : compute_autoconvolution_values(h, f_delta_x, P_val).max()

def unconstrained_loss(P_val=600):
    f_interval = (-0.25, 0.25) # interval for f(x)
    f_x_min, f_x_max = f_interval
    f_delta_x = (f_x_max - f_x_min) / P_val
    return lambda log_h : compute_autoconvolution_values(2*P_val*F.softmax(log_h, dim=0), f_delta_x, P_val).max()

###


### Projection onto the simplex ###

def projection_simplex_pytorch(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    n_features = v.shape[0]
    if n_features == 0:
        return torch.empty_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv_minus_z = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, n_features + 1, device=v.device) 
    cond = u - cssv_minus_z / ind > 0
    true_indices = torch.where(cond)[0]
    rho_idx = true_indices[-1] 
    rho = ind[rho_idx] 
    theta = cssv_minus_z[rho_idx] / rho 
    w = torch.clamp(v - theta, min=0.0)
    return w

### Algorithms ### 
def polyak_subgradient_method(h_params, loss_fn, max_iter=100000, target_loss=1.5053, print_every=100, history = {}, projection=None):
    # if history fields are not present, initialize them
    if 'loss_history' not in history:
        history['loss_history'] = []
    if 'grad_norm_history' not in history:
        history['grad_norm_history'] = []
    if 'min_loss_found' not in history:
        history['min_loss_found'] = float('inf')
    if 'best_h_params' not in history:
        history['best_h_params'] = h_params.data.clone()
        
    for i in range(max_iter):
        # Compute loss and gradient
        h_params.grad = None
        loss = loss_fn(h_params)
        grad = torch.autograd.grad(loss, h_params)[0]

        grad_norm_squared = torch.norm(grad)**2
        
        step_size = torch.clamp((loss.item() - target_loss) / grad_norm_squared, min=0.0)
        h_params.data -= step_size * grad

        # Project onto the simplex
        if projection is not None:
            with torch.no_grad():
                h_params.data = projection(h_params.data)

        history['loss_history'].append(loss.item())
        history['grad_norm_history'].append(torch.norm(grad).item())

        if loss.item() < history['min_loss_found']:
            history['min_loss_found'] = loss.item()
            history['best_h_params'] = h_params.data.clone()
        if i % print_every == 0:
            print(f"Iteration {i}: loss = {loss.item():.7f}, step_size = {step_size:.7f}, grad_norm = {history['grad_norm_history'][-1]:.7f}, min_loss_found_polyak = {history['min_loss_found']:.6f}")

    return history




def bfgs_method(h_params, loss_fn, max_iter=100000, bfgs_params={},print_every=100, history={}):
    # optimizer = torch.optim.LBFGS([h_params], lr=lr, history_size=1000)
    if 'lr' not in bfgs_params:
        bfgs_params['lr'] = 1.0
    if 'history_size' not in bfgs_params:
        bfgs_params['history_size'] = 100
    if 'ls_params' not in bfgs_params:
        bfgs_params['ls_params'] = {'c1': 0.0001, 'c2': 0.9}
    optimizer = bfgs_weak_wolfe.BFGS([h_params], lr=bfgs_params['lr'], history_size=bfgs_params['history_size'], line_search_fn=bfgs_params['line_search_fn'], ls_params=bfgs_params['ls_params'])
    # if history fields are not present, initialize them
    if 'loss_history' not in history:
        history['loss_history'] = []
    if 'grad_norm_history' not in history:
        history['grad_norm_history'] = []
    if 'min_loss_found' not in history:
        history['min_loss_found'] = float('inf')
    if 'best_h_params' not in history:
        history['best_h_params'] = h_params.data.clone()
        
    for i in range(max_iter):
        # Compute loss and gradient
        def closure(): 
            optimizer.zero_grad()
            loss = loss_fn(h_params)
            grad = torch.autograd.grad(loss, h_params)[0]
            h_params.grad = grad
            return loss
        optimizer.step(closure)
        loss = closure().item()
        grad_norm = torch.norm(h_params.grad).item()
        history['loss_history'].append(loss)
        history['grad_norm_history'].append(grad_norm)

        if loss < history['min_loss_found']:
            history['min_loss_found'] = loss
            history['best_h_params'] = h_params.data.clone()
        if i % print_every == 0:
            print(f"Iteration {i}: loss = {loss:.10f}, grad_norm = {grad_norm:.10f}, min_loss_found_bfgs = {history['min_loss_found']:.10f}")

    return history

def ntd_method(h_params, loss_fn, max_iter=100000, ntd_params={},print_every=100, history={}):
    # optimizer = torch.optim.LBFGS([h_params], lr=lr, history_size=1000)
    if 'opt_f' not in ntd_params:
        ntd_params['opt_f'] = np.inf
    if 'adaptive_grid_size' not in ntd_params:
        ntd_params['adaptive_grid_size'] = False
    if 'use_trust_region' not in ntd_params:
        ntd_params['use_trust_region'] = True
    if 's_scale_factor' not in ntd_params:
        ntd_params['s_scale_factor'] = 1e-6
    if 'verbose' not in ntd_params:
        ntd_params['verbose'] = False
    if 'grad_tol_break' not in ntd_params:
        ntd_params['grad_tol_break'] = 1e-20
    if 'max_grid_size' not in ntd_params:
        ntd_params['max_grid_size'] = 50
    if 'min_goldstein_iters' not in ntd_params:
        ntd_params['min_goldstein_iters'] = None

    optimizer = ntd.NTD([h_params],opt_f=ntd_params['opt_f'], adaptive_grid_size=ntd_params['adaptive_grid_size'], use_trust_region=ntd_params['use_trust_region'], s_scale_factor=ntd_params['s_scale_factor'], verbose=ntd_params['verbose'], grad_tol_break=ntd_params['grad_tol_break'], max_grid_size=ntd_params['max_grid_size'], min_goldstein_iters=ntd_params['min_goldstein_iters'])
    # if history fields are not present, initialize them
    if 'loss_history' not in history:
        history['loss_history'] = []
    if 'grad_norm_history' not in history:
        history['grad_norm_history'] = []
    if 'min_loss_found' not in history:
        history['min_loss_found'] = float('inf')
    if 'best_h_params' not in history:
        history['best_h_params'] = h_params.data.clone()
    print("Starting NTD method: (Note each iteration gets more expensive as time goes on)")        
    for i in range(max_iter):
        # Compute loss and gradient
        def closure(): 
            optimizer.zero_grad()
            loss = loss_fn(h_params)
            grad = torch.autograd.grad(loss, h_params)[0]
            h_params.grad = grad
            return loss
        optimizer.step(closure)
        loss = closure().item()
        grad_norm = torch.norm(h_params.grad).item()
        history['loss_history'].append(loss)
        history['grad_norm_history'].append(grad_norm)

        if loss < history['min_loss_found']:
            history['min_loss_found'] = loss
            history['best_h_params'] = h_params.data.clone()
        if i % print_every == 0:
            print(f"Iteration {i}: loss = {loss:.10f}, grad_norm = {grad_norm:.10f}, min_loss_found = {history['min_loss_found']:.10f}")

    return history



# Assume P_val, delta_x, S_target (2*P_val) are defined
# Assume compute_autoconvolution_values is defined

def solve_proximal_linearized_subproblem_full(h_k_torch: torch.Tensor,
                                              S_target_val: float,
                                              delta_x_val: float,
                                              P_val_local: int,
                                              gamma_prox: float, 
                                              solver_params: dict = {'abstol': 1e-16, 'reltol': 1e-16, 'feastol': 1e-16, 'max_iters': 2000}):
    
    h_k_np = h_k_torch.detach().cpu().numpy() # Current iterate h_k

    # CVXPY Variables
    h_cvx = cp.Variable(P_val_local, name="h") # h to solve for
    eta_cvx = cp.Variable(name="eta")          # Auxiliary variable for the max

    # --- Compute current Q_m(h_k) values and their gradients grad_Q_m(h_k) ---
    current_heights_for_Q = h_k_torch.detach().clone()
    
    # all_Q_values_torch are [Q_0(h_k), Q_1(h_k), ..., Q_{2P-1}(h_k), Q_{2P}(h_k)]
    all_Q_values_torch = compute_autoconvolution_values(current_heights_for_Q, delta_x_val, P_val_local)
    # We are interested in Q_m for m from 1 to 2P-1.
    # Q_m(h_k) corresponds to index 'm' in all_Q_values_torch if m is 0-indexed for the array,
    # or index 'm_original_idx' if m_original_idx is 1-based.
    # Let's use 0-indexed m for the Q array from here: Q_values_torch[m_knot_idx]
    # where m_knot_idx goes from 1 to 2P-1.
    
    constraints_cvxpy = []
    
    # Add linearization constraints for ALL relevant Q_m functions
    # Q_m(h) = (f_h*f_h)(t_m). Indices for knots of f*f run from t_0 to t_{2P}.
    # The values are non-trivial for t_1 to t_{2P-1}.
    # So, m (index for Q_m) runs from 1 to 2P-1.
    # This means the slice all_Q_values_torch[1:-1] contains Q_1, ..., Q_{2P-1}.
    
    for m_idx_in_relevant_slice in range(2 * P_val_local - 1): # 0 to 2P-2 for the slice
        # m_original_idx is the actual m for Q_m, running from 1 to 2P-1
        m_original_idx = m_idx_in_relevant_slice + 1 
        
        Q_m_at_h_k = all_Q_values_torch[m_original_idx].item() # Q_m(h_k)

        # Gradient of Q_m(h) w.r.t h, evaluated at h_k:
        # (grad_Q_m(h_k))_i = 2 * delta_x * (h_k)_{m-1-i}
        grad_Qm_at_h_k_np = np.zeros(P_val_local)
        for i_component in range(P_val_local):
            partner_idx = (m_original_idx - 1) - i_component
            if 0 <= partner_idx < P_val_local: # Check if partner_idx is a valid index for h_k
                 grad_Qm_at_h_k_np[i_component] = 2 * delta_x_val * h_k_np[partner_idx]
        
        # Linearization: Q_m(h_k) + grad_Q_m(h_k)^T * (h - h_k) <= eta
        # Which is: (grad_Q_m(h_k)^T * h) - eta <= (grad_Q_m(h_k)^T * h_k) - Q_m(h_k)
        
        # Constant part of linearization for constraint: Q_m(h_k) - grad_Q_m(h_k)^T * h_k
        const_part_linearization = Q_m_at_h_k - np.dot(grad_Qm_at_h_k_np, h_k_np)
        
        constraints_cvxpy.append(grad_Qm_at_h_k_np @ h_cvx + const_part_linearization <= eta_cvx)

    # Add simplex constraints for h_cvx
    constraints_cvxpy.append(cp.sum(h_cvx) == S_target_val)
    constraints_cvxpy.append(h_cvx >= 0)
    
    # Objective function for the QP subproblem
    # minimize eta + (gamma_prox / 2) * ||h - h_k||^2
    objective_cvxpy = cp.Minimize(eta_cvx + (gamma_prox / 2) * cp.sum_squares(h_cvx - h_k_np))
    
    problem_cvxpy = cp.Problem(objective_cvxpy, constraints_cvxpy)
    
    try:
        # Using OSQP as it's generally good for QPs.
        # Adjust solver parameters if needed for precision/speed.
        abstol = solver_params['abstol']
        reltol = solver_params['reltol']
        feastol = solver_params['feastol']
        max_iters = solver_params['max_iters']
        verbose = solver_params['verbose']
        if solver_params['solver'] == 'ECOS':
            problem_cvxpy.solve(solver=cp.ECOS, verbose=verbose, abstol=abstol, reltol=reltol, feastol=feastol, max_iters=max_iters)
        elif solver_params['solver'] == 'CLARABEL':
            problem_cvxpy.solve(solver=cp.CLARABEL, verbose=verbose, tol_gap_abs=abstol, tol_gap_rel=reltol, tol_feas=feastol, max_iter=max_iters)
        else:
            raise ValueError(f"Unsupported solver: {solver_params['solver']}")
    except cp.error.SolverError as e:
        print(f"CVXPY Solver error (OSQP): {e}. Subproblem might have failed.")
        # You could try another solver here as a fallback if desired.
        # e.g., problem_cvxpy.solve(solver=cp.SCS, verbose=False)
        return None, None # Indicate failure

    if problem_cvxpy.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: CVXPY problem not solved to optimality. Status: {problem_cvxpy.status}")
        if h_cvx.value is None: # Check if a solution (even suboptimal) is available
             print("No solution vector returned by solver.")
             return None, None
        print("Proceeding with potentially suboptimal/inaccurate solution from QP.")
        
    h_next_np = h_cvx.value
    # If h_next_np is None (e.g. if solver truly failed and status is like 'solver_error')
    if h_next_np is None:
        print("Critical solver failure: h_cvx.value is None.")
        return None, None

    eta_next_val = eta_cvx.value if eta_cvx.value is not None else float('nan') # Handle if eta is None
    
    return torch.tensor(h_next_np, dtype=h_k_torch.dtype, device=h_k_torch.device), eta_next_val


def prox_linear_method(h_params, loss_fn,print_every, max_iter=1000, prox_linear_params={}, history = {}):
    # if history fields are not present, initialize them
    if 'loss_history' not in history:
        history['loss_history'] = []
    if 'grad_norm_history' not in history:
        history['grad_norm_history'] = []
    if 'min_loss_found' not in history:
        history['min_loss_found'] = float('inf')
    if 'best_h_params' not in history:
        history['best_h_params'] = h_params.data.clone()
    if 'gamma_prox' not in prox_linear_params:
        prox_linear_params['gamma_prox'] = 1
    if 'abstol' not in prox_linear_params:
        prox_linear_params['abstol'] = 1e-09
    if 'reltol' not in prox_linear_params:
        prox_linear_params['reltol'] = 1e-09
    if 'feastol' not in prox_linear_params:
        prox_linear_params['feastol'] = 1e-09
    if 'max_iters' not in prox_linear_params:
        prox_linear_params['max_iters'] = 2000
    if 'solver' not in prox_linear_params:
        prox_linear_params['solver'] = 'CLARABEL'
    if 'linesearch' not in prox_linear_params:
        prox_linear_params['linesearch'] = False

    solver_params = {key: prox_linear_params[key] for key in ['abstol', 'reltol', 'feastol', 'max_iters', 'verbose', 'solver']}
    gamma_prox = prox_linear_params['gamma_prox']
    P_val_local = h_params.shape[0]
    S_target_val = 2 * P_val_local
    delta_x_val = 0.5 / P_val_local
        
    for i in range(max_iter):
        # Compute loss and gradient
        h_params.grad = None
        h_next_torch, eta_val = solve_proximal_linearized_subproblem_full(h_params, S_target_val, delta_x_val, P_val_local, gamma_prox, solver_params)
        with torch.no_grad():
            if gamma_prox > 0:
                step_length = torch.norm(h_next_torch - h_params.data).item()/gamma_prox
            else:
                step_length = torch.norm(h_next_torch - h_params.data).item()
        h_params.data = h_next_torch
        loss = loss_fn(h_params)
        grad = torch.autograd.grad(loss, h_params)[0]

        history['loss_history'].append(loss.item())
        history['grad_norm_history'].append(torch.norm(grad).item())

        if loss.item() < history['min_loss_found']:
            history['min_loss_found'] = loss.item()
            history['best_h_params'] = h_params.data.clone()

        if i % print_every == 0:
            print(f"Iteration {i}: loss = {loss.item():.10f}, step_length = {step_length:.7f}, grad_norm = {history['grad_norm_history'][-1]:.7f}, min_loss_found = {history['min_loss_found']:.10f}")

    return history



def matolcsi_kolountzakis_lp_step(
    h_k_torch: torch.Tensor,
    S_target_val: float, # Target sum for normalization, e.g., 2*P
    delta_x_val: float,
    P_val_local: int,
    t_mixing: float = 0.1, # Mixing parameter for the update
    line_search_iters: int = 0, # Number of iterations for ternary line search for t_mixing
                                # If 0, uses fixed t_mixing
    solver_params: dict = None # For the LP solver
):
    """
    Performs one step of the Matolcsi-Kolountzakis iterative LP-based method.
    1. Solves an LP to find g_0 that maximizes sum(b_j) s.t. ||f_k * g_0||_inf <= ||f_k * f_k||_inf.
    2. Normalizes g_0 to g_prime such that sum(g_prime_j) = S_target_val.
    3. Updates h_k to h_{k+1} = (1-t)*h_k + t*g_prime, possibly with line search for t.

    Args:
        h_k_torch: Current step function heights (P_val_local,). Assumed to be non-negative
                   and sum to S_target_val.
        S_target_val: The target sum for the heights vector (e.g., 2 * P_val_local).
        delta_x_val: Width of each step.
        P_val_local: Number of pieces.
        t_mixing: Fixed mixing parameter if line_search_iters is 0.
        line_search_iters: If > 0, number of iterations for ternary line search for t_mixing.
        solver_params: Dictionary of parameters for the CVXPY LP solver.
                       Example: {'solver': 'ECOS', 'verbose': False, 'abstol': 1e-8, ...}

    Returns:
        h_next_torch: The updated heights vector (P_val_local,).
                      Returns original h_k_torch if LP fails or no improvement direction.
        actual_L_h_next: The ||h_next * h_next||_inf value.
        lp_objective_value: The sum(b_j) achieved by the LP.
    """
    if solver_params is None:
        # Default solver parameters for the LP
        solver_params = {'solver': 'ECOS', 'verbose': False, 
                         'abstol': 1e-8, 'reltol': 1e-8, 'feastol': 1e-8, # ECOS specific names
                         'max_iters': 200}
        


    a_k_np = h_k_torch.detach().cpu().numpy() # Current heights f_k = (a_j)

    # 1. Compute C_max_k = ||f_k * f_k||_infinity
    f_k_f_k_values = compute_autoconvolution_values(h_k_torch, delta_x_val, P_val_local)
    # Max over relevant knots (t_1 to t_{2P-1})
    # f_k_f_k_values has length 2P+1. Indices 1 to 2P-1 (exclusive end)
    C_max_k = torch.max(f_k_f_k_values[1:-1]).item() 

    # CVXPY Variables for the LP
    b_cvx = cp.Variable(P_val_local, name="b_coeffs", nonneg=True) # b_j >= 0

    constraints_lp = []

    # Add constraints: -(C_max_k) <= (f_k * g_0)(t_m) <= C_max_k
    # where g_0 is represented by b_cvx. (f_k * g_0)(t_m) = sum_i (delta_x * a_k_{m-1-i}) * b_i
    for m_original_idx in range(1, 2 * P_val_local): # m from 1 to 2P-1
        coeff_vector_m_np = np.zeros(P_val_local)
        for i_component_b in range(P_val_local): # i_component_b is the index for b_cvx
            # Coefficient for b_cvx[i_component_b] in the m-th cross-convolution sum
            # Term is a_k_partner * b_cvx[i_component_b] where partner index is (m-1-i_component_b)
            partner_idx_for_a = (m_original_idx - 1) - i_component_b
            if 0 <= partner_idx_for_a < P_val_local:
                coeff_vector_m_np[i_component_b] = delta_x_val * a_k_np[partner_idx_for_a]
        
        f_k_g_0_tm_expr = coeff_vector_m_np @ b_cvx
        
        constraints_lp.append(f_k_g_0_tm_expr <= C_max_k)
        constraints_lp.append(f_k_g_0_tm_expr >= -C_max_k) 
        # The >= -C_max_k is important as cross-convolution can be negative if not all a_k, b_j are positive,
        # but here a_k are from h_k_torch (non-neg) and b_cvx is nonneg=True. So f_k_g_0_tm_expr will be non-negative.
        # Thus, >= -C_max_k is auto-satisfied if C_max_k >=0. Still, good to include for robustness.

    # Objective function: maximize sum(b_j)
    objective_lp = cp.Maximize(cp.sum(b_cvx))
    problem_lp = cp.Problem(objective_lp, constraints_lp)
    
    lp_objective_value = -float('inf') # Default if LP fails

    try:
        solver_name = solver_params.get('solver', 'ECOS').upper()
        cvxpy_params = {k: v for k, v in solver_params.items() if k != 'solver'}
        
        # Adjust tolerance names for Clarabel if used
        if solver_name == 'CLARABEL':
            if 'abstol' in cvxpy_params: cvxpy_params['tol_gap_abs'] = cvxpy_params.pop('abstol')
            if 'reltol' in cvxpy_params: cvxpy_params['tol_gap_rel'] = cvxpy_params.pop('reltol')
            if 'feastol' in cvxpy_params: cvxpy_params['tol_feas'] = cvxpy_params.pop('feastol')
            if 'max_iters' in cvxpy_params: cvxpy_params['max_iter'] = cvxpy_params.pop('max_iters') # Clarabel uses max_iter

        problem_lp.solve(solver=getattr(cp, solver_name), **cvxpy_params)
        lp_objective_value = problem_lp.value if problem_lp.value is not None else -float('inf')

    except Exception as e: # Broader exception catch for solver issues
        print(f"LP Solver failed with {solver_params.get('solver', 'ECOS')}: {e}. Problem status: {problem_lp.status}")
        # No improvement if LP fails, return original h_k
        current_L_val = torch.max(compute_autoconvolution_values(h_k_torch, delta_x_val, P_val_local)[1:-1]).item()
        return h_k_torch.clone(), current_L_val, lp_objective_value

    if problem_lp.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: LP problem not solved to optimality by {solver_params.get('solver', 'ECOS')}. Status: {problem_lp.status}")
        if b_cvx.value is None:
            current_L_val = torch.max(compute_autoconvolution_values(h_k_torch, delta_x_val, P_val_local)[1:-1]).item()
            return h_k_torch.clone(), current_L_val, lp_objective_value
        # Else, proceed with potentially suboptimal g_0
    
    g_0_np = b_cvx.value
    if g_0_np is None:
        print("Critical LP solver failure: b_cvx.value is None.")
        current_L_val = torch.max(compute_autoconvolution_values(h_k_torch, delta_x_val, P_val_local)[1:-1]).item()
        return h_k_torch.clone(), current_L_val, lp_objective_value
        
    g_0_torch = torch.tensor(g_0_np, dtype=h_k_torch.dtype, device=h_k_torch.device)

    # 2. Normalize g_0 to g_prime so that sum(g_prime_j) = S_target_val
    sum_b_j = torch.sum(g_0_torch).item()
    if sum_b_j > 1e-16: # Avoid division by zero or very small sum
        g_prime_torch = (S_target_val / sum_b_j) * g_0_torch
    else:
        # If LP solution is near zero, g_prime can't be meaningfully normalized to S_target.
        # This might happen if C_max_k is very restrictive (e.g. h_k is already great or C_max_k is tiny)
        # Default to no change in this case or use h_k as g_prime.
        print("  LP resulted in sum(b_j) near zero. Using h_k as g_prime (no improvement direction).")
        g_prime_torch = h_k_torch.clone()

    # 3. Determine mixing parameter t and update h
    chosen_t_mixing = t_mixing
    if line_search_iters > 0:
        # Define the objective for line search: L(t_mix) = || ((1-t_mix)h_k + t_mix*g_prime)^2 ||_inf
        def L_of_t_mix_func(t_mix_val):
            if not (0.0 <= t_mix_val <= 1.0): return float('inf')
            h_mixed = (1 - t_mix_val) * h_k_torch + t_mix_val * g_prime_torch
            # h_mixed should be non-negative here as t_mix_val is in [0,1]
            # and h_k_torch, g_prime_torch are non-negative.
            autoconv_mixed = compute_autoconvolution_values(h_mixed, delta_x_val, P_val_local)
            return torch.max(autoconv_mixed[1:-1]).item()

        # Ternary search for t_mix_val
        t_left, t_right = 0.0, 1.0
        for _ in range(line_search_iters):
            if abs(t_right - t_left) < 1e-7: break
            m1 = t_left + (t_right - t_left) / 3
            m2 = t_right - (t_right - t_left) / 3
            if L_of_t_mix_func(m1) < L_of_t_mix_func(m2):
                t_right = m2
            else:
                t_left = m1
        chosen_t_mixing = (t_left + t_right) / 2
        # print(f"  Line search chose t_mixing = {chosen_t_mixing:.4f}")

    h_next_torch = (1 - chosen_t_mixing) * h_k_torch + chosen_t_mixing * g_prime_torch
    
    # Ensure h_next is non-negative and normalized (should be if inputs are and t is in [0,1])
    h_next_torch = torch.clamp(h_next_torch, min=0.0) # Safety for numerical precision
    current_sum_h_next = torch.sum(h_next_torch).item()
    if abs(current_sum_h_next) > 1e-9 : # Avoid division by zero
         h_next_torch = (S_target_val / current_sum_h_next) * h_next_torch # Re-normalize
    else: # h_next became zero vector, should not happen with t in [0,1] unless h_k and g_prime were zero
         h_next_torch = torch.full_like(h_k_torch, S_target_val / P_val_local) # Reinitialize to uniform


    actual_L_h_next = torch.max(compute_autoconvolution_values(h_next_torch, delta_x_val, P_val_local)[1:-1]).item()
    
    return h_next_torch, actual_L_h_next, lp_objective_value



def lp_method(h_params, loss_fn,print_every, max_iter=1000, lp_params={}, history = {}):
    # if history fields are not present, initialize them
    if 'loss_history' not in history:
        history['loss_history'] = []
    if 'grad_norm_history' not in history:
        history['grad_norm_history'] = []
    if 'min_loss_found' not in history:
        history['min_loss_found'] = float('inf')
    if 'best_h_params' not in history:
        history['best_h_params'] = h_params.data.clone()
    if 'abstol' not in lp_params:
        lp_params['abstol'] = 1e-09
    if 'reltol' not in lp_params:
        lp_params['reltol'] = 1e-09
    if 'feastol' not in lp_params:
        lp_params['feastol'] = 1e-09
    if 'max_iters' not in lp_params:
        lp_params['max_iters'] = 2000
    if 'solver' not in lp_params:
        lp_params['solver'] = 'CLARABEL'
    if 'line_search_iters' not in lp_params:
        lp_params['line_search_iters'] = 10

    solver_params = {key: lp_params[key] for key in ['abstol', 'reltol', 'feastol', 'max_iters', 'verbose', 'solver']}
    P_val_local = h_params.shape[0]
    S_target_val = 2 * P_val_local
    delta_x_val = 0.5 / P_val_local
        
    for i in range(max_iter):
        # Compute loss and gradient
        h_params.grad = None
        h_next_torch, _, _ = matolcsi_kolountzakis_lp_step(h_params, S_target_val, delta_x_val, P_val_local, t_mixing=1, line_search_iters=lp_params['line_search_iters'], solver_params=solver_params)
        with torch.no_grad():
            step_length = torch.norm(h_next_torch - h_params.data).item()
        h_params.data = h_next_torch
        loss = loss_fn(h_params)
        grad = torch.autograd.grad(loss, h_params)[0]

        history['loss_history'].append(loss.item())
        history['grad_norm_history'].append(torch.norm(grad).item())

        if loss.item() < history['min_loss_found']:
            history['min_loss_found'] = loss.item()
            history['best_h_params'] = h_params.data.clone()

        if i % print_every == 0:
            print(f"Iteration {i}: loss = {loss.item():.10f}, step_length = {step_length:.7f}, grad_norm = {history['grad_norm_history'][-1]:.7f}, min_loss_found = {history['min_loss_found']:.10f}")

    return history