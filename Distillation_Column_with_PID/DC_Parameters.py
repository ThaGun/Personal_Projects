''' Kinetic Parameter calculation '''

class ParametersCalc:
    
    def __init__(self, Number_of_stages, Flow_stages):
        self.f_s = Flow_stages
        self.N = Number_of_stages
        self.alpha = 2.317           # Relative volatility

    def MoleFractionOfVapor(self, MoleFracOfL):
        MoleFracOfV = (self.alpha * MoleFracOfL) / 1 + (self.alpha - 1) * MoleFracOfL
        return MoleFracOfV
    
    def RateOfFlowOfLiquid(self, LiquidHoldup):
        RateOfFLowOfL = 3.0 * ((LiquidHoldup) ** 3/2)
        return RateOfFLowOfL