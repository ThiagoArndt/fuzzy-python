import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Cria as variáveis do problema
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
tempo = ctrl.Antecedent(np.arange(0, 11, 1), 'tempo')
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')



comida['insossa'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['saborosa'] = fuzz.gaussmf(comida.universe, 10,3)


servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['excelente'] = fuzz.gaussmf(servico.universe, 10,3)

tempo['demorado'] = fuzz.trimf(tempo.universe,[0, 0, 13])
tempo['mediano'] = fuzz.trapmf(tempo.universe,[0, 13,15, 25])
tempo['rapido'] = fuzz.trimf(tempo.universe, [15, 25, 25])

gorjeta['nenhuma'] = fuzz.trimf(gorjeta.universe, [0, 0, 13])
gorjeta['pouca'] = fuzz.trapmf(gorjeta.universe, [0, 13,15, 25])
gorjeta['generosa'] = fuzz.trimf(gorjeta.universe, [15, 25, 25])




rule1 = ctrl.Rule(comida['insossa'] & servico['ruim'], gorjeta['pouca'])
rule2 = ctrl.Rule(comida['saborosa'] & servico['excelente'], gorjeta['generosa'])
rule3 = ctrl.Rule(tempo['demorado'], gorjeta['nenhuma'])
rule4 = ctrl.Rule(tempo['mediano'] | tempo['rapido'], gorjeta['pouca'])
rule5 = ctrl.Rule(tempo['mediano'] | tempo['rapido'], gorjeta['generosa'])

gorjeta_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
gorjeta_simulador = ctrl.ControlSystemSimulation(gorjeta_ctrl)


gorjeta_simulador.input['comida'] = 3.5
gorjeta_simulador.input['tempo'] = 3.5
gorjeta_simulador.input['servico'] = 9.4

# Computando o resultado
gorjeta_simulador.compute()
print(gorjeta_simulador.output['gorjeta'])

comida.view(sim=gorjeta_simulador)
servico.view(sim=gorjeta_simulador)
gorjeta.view(sim=gorjeta_simulador)
plt.pause(100)