from problem.basic_problem import Basic_Problem
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
try:
    import matlab.engine
except ImportError:
    raise ImportError("The 'matlab.engine' package is not installed. Please install the MATLAB Engine API for Python by following the official MATLAB documentation.")


dir = os.getcwd()
eng = matlab.engine.start_matlab()
print("Matlab Running...")
eng.cd(dir)
eng.addpath('data_files') # add matlab function

class UAV_Basic_Problem(Basic_Problem):
    def __init__(self):
        self.terrain_model = None
        self.FES = 0
        self.optimum = None
        self.problem_id = None
        self.dim = None
        self.lb = None
        self.ub = None

    def __str__(self):
        return f"Terrain {self.problem_id}"

    def __boundaries__(self):
        model = self.terrain_model

        nVar = model['n']

        # Initialize the boundaries dictionaries
        VarMin = {'x': model['xmin'], 'y': model['ymin'], 'z': model['zmin'],
                  'r': 0, 'psi': -np.pi / 4, 'phi': None}
        VarMax = {'x': model['xmax'], 'y': model['ymax'], 'z': model['zmax'],
                  'r': None, 'psi': np.pi / 4, 'phi': None}

        # Calculate the radial distance range based on the model's start and end points
        distance = np.linalg.norm(np.array(model['start']) - np.array(model['end']))
        VarMax['r'] = 2 * distance / nVar

        # Inclination (elevation) limits (angle range is pi/4)
        AngleRange = np.pi / 4
        VarMin['psi'] = -AngleRange
        VarMax['psi'] = AngleRange

        # Azimuth (phi)
        dirVector = np.array(model['end']) - np.array(model['start'])
        phi0 = np.arctan2(dirVector[1], dirVector[0])
        VarMin['phi'] = (phi0 - AngleRange).item()
        VarMax['phi'] = (phi0 + AngleRange).item()

        # Lower and upper Bounds of velocity
        alpha = 0.5
        VelMax = {'r': alpha * (VarMax['r'] - VarMin['r']),
                  'psi': alpha * (VarMax['psi'] - VarMin['psi']),
                  'phi': alpha * (VarMax['phi'] - VarMin['phi'])}
        VelMin = {'r': -VelMax['r'],
                  'psi': -VelMax['psi'],
                  'phi': -VelMax['phi']}

        # Create bounds by stacking both position and velocity limits
        bounds = np.array([
            [VarMin['r'], VarMax['r']],
            [VarMin['psi'], VarMax['psi']],
            [VarMin['phi'], VarMax['phi']],
        ])
        # Since we are interested in r, phi, psi for nVar points in each item of the population
        bounds = np.tile(bounds, (int(nVar), 1))

        # Assign the 0th column to self.lb and the 1st column to self.ub
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

    def spherical_to_cart_vec(self, solve):
        # solve: 2D [NP, 3 * n]
        pass

    def DistP2S(self, xs, a, b):
        # xs: 1D array [2], a: 2D array [2, NP], b: 2D array [2, NP]
        pass

    def func(self, x):
        raise NotImplementedError

class Terrain(UAV_Basic_Problem):
    def __init__(self, terrain_model, problem_id):
        super(Terrain, self).__init__()
        self.terrain_model = terrain_model
        self.dim = 3 * terrain_model['n']
        self.problem_id = problem_id
        self.optimum = None
        self.__boundaries__()
        self.SphCost = eng.CreateSphCost(self.terrain_model)

    def func(self, x):
        # x: 2D [NP, 3 * n] dim = 3 * n
        return np.array([eng.feval(self.SphCost, temp_x[:, None], nargout = 4) for temp_x in x])

class UAV_Dataset(Dataset):
    def __init__(self, data, batch_size = 1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(train_batch_size=1,
                     test_batch_size = 1,
                     dv = 5.0,
                     j_pen = 1e4,
                     difficulty='easy'):
        mat_file = "problem/data_files/Model56.mat"
        mat_data = eng.load(mat_file)
        model_data = mat_data['Model']
        # pkl_file = "problem/UAE_terrain_data/Model56.pkl"
        # with open(pkl_file, 'rb') as f:
        #     model_data = pickle.load(f)
        func_id = range(56) # 56
        train_set = []
        test_set = []
        for id in func_id:
            terrain_data = model_data[id]
            terrain_data['n'] = dv
            terrain_data['J_pen'] = j_pen
            instance = Terrain(terrain_data, id + 1)
            train_set.append(instance)
            test_set.append(instance)
        return UAV_Dataset(train_set, train_batch_size), UAV_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        if self.batch_size < 2:
            return self.data[self.index[item]]
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'UAV_Dataset'):
        return UAV_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)

if __name__ == "__main__":
    mat_file = "problem/data_files/Model56.mat"
    mat_data = eng.load(mat_file)
    model_data = mat_data['Model']
    func_id = 4
    terrain_data = model_data[func_id]
    terrain_data['n'] = 10.0
    terrain_data['J_pen'] = 1e4
    fun5 = Terrain(terrain_data, 5)

    x =  [
        [
            102.787067274924, -0.588464384157333, 0.523312419417429, 8.98171798143932,
            0.0707168743046490, 1.49658000170548, 80.6110279168768, -0.195561226893039,
            0.574004754481592, 141.203633522607, -0.369189314002375, 0.551542337270883,
            179.697141403396, 0.604469599761478, 0.521125312856793, 112.889900716559,
            -0.153310866363274, 0.599568876578121, 156.479010612955, 0.515890617157642,
            0.761018950795744, 158.169662571115, -0.159932706752021, 0.242214520089881,
            94.8344807895242, -0.650786093007384, 1.46528689191431, 20.9142143722353,
            -0.173403956357546, 0.724803312220412
        ],
        [
            102.787067274924, -0.588464384157333, 0.523312419417429, 8.98171798143932,
            0.0707168743046490, 1.49658000170548, 80.6110279168768, -0.195561226893039,
            0.574004754481592, 141.203633522607, -0.369189314002375, 0.551542337270883,
            179.697141403396, 0.604469599761478, 0.521125312856793, 112.889900716559,
            -0.153310866363274, 0.599568876578121, 156.479010612955, 0.515890617157642,
            0.761018950795744, 158.169662571115, -0.159932706752021, 0.242214520089881,
            94.8344807895242, -0.650786093007384, 1.46528689191431, 20.9142143722353,
            -0.173403956357546, 0.724803312220412
        ]
    ]
    np.set_printoptions(10)
    x_np = np.array(x)
    cost = fun5.func(x_np)
    print(cost)
    '''
    Matlab Running...
    [[ 1272.6309300376 90008.5486544402   425.826835986    417.6340855462]
     [ 1272.6309300376 90008.5486544402   425.826835986    417.6340855462]]
    '''
    # add weight
    Cost = 5 * cost[:, 0] + 1 * cost[:, 1] + 10 * cost[:, 2] + 1 * cost[:, 3]
    print(Cost)
    '''
    [101047.6057500338 101047.6057500338]
    '''