'''
python module which defines model classes
'''
import pyomo.environ as pyo
import numpy as np
np.set_printoptions(2)

def Objective_mls(mod):
    '''
    multivariate response least squares objective
    '''
    M = mod.M
    P = mod.P
    T = mod.T

    return(
        pyo.quicksum( mod.Q[m][p1,p2] * mod.beta[m,p1] * mod.beta[m,p2] for
                 m in range(M) for p1 in range(P) for p2 in range(P) ) +
        pyo.quicksum( mod.R[m][p] * mod.beta[m,p] for m in range(M) for p in range(P) ) +
        pyo.quicksum( mod.C )
    )

def Objective_mls_shrinkl1(mod):
    M = mod.M
    P = mod.P
    T = mod.T

    return(
        pyo.quicksum( mod.Q[m][p1,p2] * mod.beta[m,p1] * mod.beta[m,p2] for
                 m in range(M) for p1 in range(P) for p2 in range(P) ) +
        pyo.quicksum( mod.R[m][p] * mod.beta[m,p] for m in range(M) for p in range(P) ) +
        pyo.quicksum( mod.C ) +
        pyo.quicksum( mod.param_gamma*(mod.beta_tilde[m,p,0]+mod.beta_tilde[m,p,1]) for
                      p in range(P) for m in range(M) )
    )

def Objective_mls_shrinkl2(mod):
    M = mod.M
    P = mod.P
    T = mod.T

    return(
        pyo.quicksum( mod.Q[m][p1,p2] * mod.beta[m,p1] * mod.beta[m,p2] for
                 m in range(M) for p1 in range(P) for p2 in range(P) ) +
        pyo.quicksum( mod.R[m][p] * mod.beta[m,p] for m in range(M) for p in range(P) ) +
        pyo.quicksum( mod.C ) +
        pyo.quicksum( mod.param_lambda*mod.beta[m,p]*mod.beta[m,p] for
                      p in range(P) for m in range(M) ) +
        pyo.quicksum( -2*mod.param_lambda*mod.beta_aux[p]*mod.beta[m,p] for
                      p in range(P) for m in range(M) ) +
        pyo.quicksum( mod.param_lambda*mod.beta_aux[p]*mod.beta_aux[p] for
                      p in range(P) for m in range(M) )
    )

class SIS:
    '''
    simultaneous integer selection class
    '''
    def __init__(self, x, y, bigM=100, shrinkage=None, factor=2, nlambda=10, ngamma=10
    ):
        '''
        x (list lenth M):  arrays of predictor variables
        y (list length M): arrays of response variables
        bigM (float):      bigM parameter value
        '''
        self.shrink = shrinkage
        self.factor = factor
        self.nlambda = nlambda
        self.ngamma = ngamma

        M = len(x)
        T = [_x.shape[0] for _x in x]
        _,P = x[0].shape

        # pyomo optimisation model
        model = pyo.ConcreteModel()
        '''
        model properties
        '''
        # number of regression models to fit
        model.M = M
        # number of input predictors
        model.P = P
        # number of observations
        model.T = T

        # ordered set of predictor indices
        model.oset_P = pyo.Set( name = "indices: predictors",
                                within = pyo.NonNegativeIntegers,
                                initialize = list(range(P)),
                                ordered = True
        )
        # ordered set of model indices
        model.oset_M = pyo.Set( name = "indices: models",
                                within = pyo.NonNegativeIntegers,
                                initialize = range(M),
                                ordered = True
        )

        '''
        model parameters
        '''
        # big M 
        model.param_bigM = 100
        # k: model sparsity 
        model.param_k = pyo.Param( default=1,
                                   doc="maximum number of predictors to allow into models",
                                   within = pyo.PositiveIntegers,
                                   mutable = True
        )
        
        # lambda: simultaneous shrinkage parameter (ridge shrinkage)
        model.param_lambda = pyo.Param( initialize=0,
                                        doc="ridge-like simultaneous shrinkage parameter",
                                        within = pyo.NonNegativeReals,
                                        mutable = True
        )
        # gamma: l1 parameter
        model.param_gamma = pyo.Param( initialize=0,
                                       doc="lasso-like simultaneous shrinkage parameter",
                                       within = pyo.NonNegativeReals,
                                       mutable = True
        )

        '''
        model variables
        '''
        # beta: regression coefficients (continuous variables)
        model.beta = pyo.Var( model.oset_M, model.oset_P, domain=pyo.Reals )
        # z: selected variable indicator (binary variables)
        model.z = pyo.Var( model.oset_P, domain=pyo.Boolean )
        # betaaux
        model.beta_aux = pyo.Var( model.oset_P, domain=pyo.Reals )
        # betatilde
        model.beta_tilde = pyo.Var( model.oset_M, model.oset_P, [0,1], domain=pyo.PositiveReals )
        
        '''
        model arrays
        '''
        model.Q = [ x[m].T.dot(x[m]) for m in range(M) ]
        model.R = [ -2*y[m].dot(x[m]) for m in range(M) ]
        model.C = [ y[m].T.dot(y[m]) for m in range(M) ]

        # objective, simultaneous OLS
        model.OBJ = pyo.Objective( rule=Objective_mls, sense=pyo.minimize )

        # objective, l2 shrinkage
        model.OBJ_shrinkl2 = pyo.Objective( rule=Objective_mls_shrinkl2,
                                            sense=pyo.minimize
        )
        model.OBJ_shrinkl2.deactivate()
        # objective, l1 shrinkage
        model.OBJ_shrinkl1 = pyo.Objective( rule=Objective_mls_shrinkl1,
                                            sense=pyo.minimize
        )
        model.OBJ_shrinkl1.deactivate()
            
        '''
        model constraints
        '''
        # sparsity: total number of selected predictors (across M models)
        model.constr_sparsity = pyo.Constraint(
            expr = pyo.quicksum(model.z[p] for p in range(P)) <= model.param_k
        )
        
        # big M above
        model.constr_Mpos = pyo.Constraint(
            model.oset_M, model.oset_P,
            rule = lambda mod,m,p: mod.beta[m,p] <= mod.param_bigM*mod.z[p]
        )
        
        # big M below
        model.constr_Mneg = pyo.Constraint(
            model.oset_M, model.oset_P,
            rule = lambda mod,m,p: -mod.param_bigM*model.z[p] <= mod.beta[m,p]
        )


        # l1 shrink: additional constraints
        model.constr_betatildep = pyo.Constraint(
            model.oset_M, model.oset_P,
            rule = lambda mod,m,p: mod.beta[m,p] - mod.beta_aux[p] <= mod.beta_tilde[m,p,0] - mod.beta_tilde[m,p,1]
        )
        model.constr_betatildep.deactivate()
        model.constr_betatilden = pyo.Constraint(
            model.oset_M, model.oset_P,
            rule = lambda mod,m,p: mod.beta[m,p] - mod.beta_aux[p] >= mod.beta_tilde[m,p,0] - mod.beta_tilde[m,p,1]
        )
        model.constr_betatilden.deactivate()

        # asssign the pyomo model to the class
        self.model = model
        
    def get_beta_array(self):
        solution = np.array(
            [[self.model.beta[m,p].value for p in range(self.model.P)] for m in range(self.model.M)]
        )

        return( solution )

    def get_current_selected(self):
        return( [p for p in range(self.model.P) if self.model.z[p].value == 1 ] )


    def get_current_notselected(self):
        return( [p for p in range(self.model.P) if self.model.z[p].value == 0 ] )


    def shrink_l2(self, g, solution, opt):
        
        # deactivate the standard objective
        self.model.OBJ.deactivate()
        
        # activate the objective with shrinkage
        self.model.OBJ_shrinkl2.activate()

        # calculate the ridge shrinkage parameter values
        vals_lambda = np.exp(
            np.linspace(np.log(1e-6), np.log(1), num=self.nlambda, endpoint=True))*self.factor*g

        for i_lambda in range(1, self.nlambda):
            self.model.param_lambda = vals_lambda[i_lambda]
            results = opt.solve( self.model )
            solution.append( self.get_beta_array() )

    def shrink_l1(self, g, solution, opt):
        # deactivate the standard objective
        self.model.OBJ.deactivate()
        
        # activate the objective with shrinkage
        self.model.OBJ_shrinkl1.activate()

        # activate the constraints
        self.model.constr_betatildep.activate()
        self.model.constr_betatilden.activate()
        #self.model.pprint()

        vals_gamma = np.exp(
            np.linspace(np.log(1e-6), np.log(1), num=self.ngamma, endpoint=True))*self.factor*g
        

        for i_gamma in range(1, self.ngamma):
            self.model.param_gamma = vals_gamma[i_gamma]
            results = opt.solve( self.model )

            solution.append( self.get_beta_array() )
    
class SBS(SIS):
    '''
    the Simultaneous Best Subset (SBS) model

    select the best predictors simultaneously amongst multiple linear regression models
    '''

    def solve(self, solver='gurobi'):
        return( self.solve_kmax(self.model.P, solver) )

    def solve_kmax(self, kmax, solver='gurobi'):

        sol = []
        for k in range(1, kmax+1):
            sol.append( self.solve_k(k, solver) )

        return( np.array(sol) )

    def solve_k(self, k, solver='gurobi'):
        opt = pyo.SolverFactory( solver )
        self.model.param_k = k

        solution = []
        results = opt.solve( self.model )
        solution.append( self.get_beta_array() )

        if self.shrink is None:
            return( solution[0] )
        
        if self.shrink == "l2":
            self.shrink_l2( pyo.value( self.model.OBJ ), solution, opt )
            
            return( np.array(solution) )

        if self.shrink == "l1":
            self.shrink_l1( pyo.value( self.model.OBJ ), solution, opt )
            
            return( np.array(solution) )


class SFS(SIS):
    '''
    Simultaneous Forward Selection (SFS)
    '''
    def solve(self, solver='gurobi'):
        opt = pyo.SolverFactory( solver )

        solution = []

        for k in range(1, self.model.P+1):

            self.model.param_k = k
            results = opt.solve( self.model )
            solution.append( self.get_beta_array() )
            
            tmp_selected = self.get_current_selected()
            
            for p in tmp_selected:
                self.model.z[p].fix(True)

        return( np.array(solution) )

class SBE(SIS):
    '''
    Simultaneous Backward Elimination (SBE)
    '''

    def solve(self, solver='gurobi'):
        opt = pyo.SolverFactory( solver )
    
        solution = []
        tmp_eliminated = []
        
        for k in range(1,self.model.P+1):
            
            self.model.param_k = (self.model.P - k + 1)
            results = opt.solve( self.model )
            solution.append( self.get_beta_array() )

            tmp_eliminated = self.get_current_notselected()
            for p in tmp_eliminated:
                self.model.z[p].fix(0)
            
        return( np.array(solution) )
