# Python code for the paper : "Resampling and denoising deep learning algorithms impact on radiomics in brain metastases MRI"
# coding: utf-8
# @author: aureliencd 

from scipy import stats
import pandas as pad
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
import math
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr

#custom_palette = [sns.xkcd_rgb["medium green"], sns.xkcd_rgb["pale red"], "blue", "orange", "blue","yellow", "purple"]
custom_paletteGeneral = [sns.xkcd_rgb["windows blue"], sns.xkcd_rgb["pale red"], sns.xkcd_rgb["medium green"], "orange", "blue","yellow", "purple"]
sns.set_palette(custom_paletteGeneral)



########################################
###        Path and dataframe        ###
########################################

### Resampling
Path = 'xxx'
df = pad.read_excel("xxx", header=0)

### Denoising
Path = 'xxx'
df = pad.read_excel("xxx", header=0)

## IA vs input
df = pad.read_excel("xxx", header=0)

##PSNR_SSIM
df = pad.read_excel("xxx", header=0)



																			#####################################
																			### Analysis of Algorithm effect  ###
																			#####################################


### Analyse PSNR et SSIM ###
results = stats.ttest_rel(df[df.columns[3]][:int(len(df)/2)], df[df.columns[3]][int(len(df)/2):])
ax = sns.boxplot(x='groupe', y=df[df.columns[3]], data=df, showfliers = False) 
figure = ax.get_figure()
figure.savefig(Path +  str(df.columns[3]), dpi=400)

results = stats.ttest_rel(df[df.columns[6]], df[df.columns[7]])
ax = sns.boxplot(x='Groupe', y=df[df.columns[3]], data=df, showfliers = False) 
figure = ax.get_figure()
figure.savefig(Path + str(df.columns[3]), dpi=400)
results[1]
### Analyse PSNR et SSIM ###


def saveTtestAnalysis(df):
	
	### significant
	y=3 
	for i in range(104): 
		results = stats.ttest_rel(df[df.columns[y]][:int(len(df)/2)], df[df.columns[y]][int(len(df)/2):])  ### pour test non app = stats.ttest_ind
		ax = sns.boxplot(x='Groupe', y=df[df.columns[y]], data=df, showfliers = False) 
		figure = ax.get_figure()
		if results[1] < 0.05:
			figure.savefig(Path + "ttest/" +"/t-test__" + str(df.columns[y]), dpi=400)
			print("Pour la variable " + str(df.columns[y] + " la valeur de p concernant la différence entre les deux images est de " + str(results[1]))) 
		figure.clear()
		y+=1

	### non significant
	y=3 
	for i in range(104): 
		results = stats.ttest_rel(df[df.columns[y]][:int(len(df)/2)], df[df.columns[y]][int(len(df)/2):])  ### pour test non app = stats.ttest_ind
		ax = sns.boxplot(x='Groupe', y=df[df.columns[y]], data=df, showfliers = False) 
		figure = ax.get_figure()
		if results[1] > 0.05:
			figure.savefig(Path + "ttest/" +"/PAS SIGNIFICATIF/t-test__" + str(df.columns[y]), dpi=400)
			print("Pour la variable " + str(df.columns[y] + " la valeur de p est supérieur à 0.05, et est égale à : " + str(results[1])))
		figure.clear()
		y+=1

saveTtestAnalysis(df)





									#############################################################################################
									### 	Correlation study between features values before and after algorithm application  ###
									### 	which radiomics features are stable after algorithm application        		   	  ###
									#############################################################################################

									#####################################################################################
									### Etude corrélation entre facteurs avant et après application de l'aglo d'IA    ###
									### 	est-ce que le paramètre de radiomic et stable après IA        		   	  ###
									#####################################################################################


def saveCorrelationFeatures(df):
	y=3 
	for i in range(104): 
		results = pearsonr(df[df.columns[y]][:int(len(df)/2)], df[df.columns[y]][int(len(df)/2):])
		ax = sns.regplot(x=df[df.columns[y]][:int(len(df)/2)], y=df[df.columns[y]][int(len(df)/2):], data=df)
		ax.set(xlabel="Origin " + str(df.columns[y]), ylabel="IA " + str(df.columns[y]))
		ax.text(0.05, 0.9, "R²=" + str(round(results[0]*results[0],3)), horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
		figure = ax.get_figure()
		figure.savefig(Path + "Corrélation features/Effet_algo__" + str(df.columns[y]), dpi=400)
		print("Pour la variable " + str(df.columns[y] + " la valeur de p est inférieur à 0.05, et R² est égal à : " + str(round(results[0]*results[0],3))))
		figure.clear()
		y+=1

saveCorrelationFeatures(df)


def PrintCorrelationFeatures(df):
	y=3
	for i in range(104): 
		results = pearsonr(df[df.columns[y]][:int(len(df)/2)], df[df.columns[y]][int(len(df)/2):])
		if results[1] < 0.05:
			print("Pour la variable " + str(df.columns[y] + " la valeur de p est égal à : " + str(results[1])))
		elif results[1] > 0.05:
			print("Pour la variable " + str(df.columns[y] + " la valeur de p est égal à : " + str(results[1])))
		y+=1

PrintCorrelationFeatures(df)




###########################################################################################
### 																   		   	        ###
### Etude Concordance Correlation Coefficient (CCC) cf article Philippe Lambin et al    ###
### 																   		   	        ###
###########################################################################################


def ccc(x,y):
	sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]      
	rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)      
	return rhoc  



### Toutes loc ###
### Stockage des valeurs de CCC dans des listes ###
FeatursAllLocList = []
CCCAllLocValue = []

y=3 
for i in range(104): 
	result = ccc(df[df.columns[y]][:int(len(df)/2)].to_numpy(), df[df.columns[y]][int(len(df)/2):].to_numpy())
	FeatursAllLocList.append(str(df.columns[y]))
	CCCAllLocValue.append(round(result,3))
	if result > 0.85:
		print("Pour la variable " + str(df.columns[y]) + " le CCC est de : " + str(round(result,3)))
	y+=1

plt.figure(figsize=(5,25))
plt.axvline(0.85, 0,1, linewidth=3, color='b')
ax = sns.barplot(x=CCCAllLocValue, y=FeatursAllLocList)
ax.set(xlabel="CCC ")
plt.show()
### Toutes loc ###



### Save CCC ###
def saveCCCRadiomic(CCC, Featurs):
	df_CustomPalette = pad.DataFrame(CCC, columns = ['CCC'])
	df_CustomPalette['Featurs'] = Featurs
	custom_palette = {}
	for q in set(df_CustomPalette.Featurs):
		val = df_CustomPalette[df_CustomPalette.Featurs == q].CCC
		if val.values < 0.85:
			custom_palette[q] = sns.xkcd_rgb["pale red"]
		else:
			custom_palette[q] = sns.xkcd_rgb["windows blue"]
	plt.figure(figsize=(5,42))
	plt.axvline(0.85, 0,1, linewidth=3, color='b')
	ax = sns.barplot(x=CCC, y=Featurs, palette=custom_palette)
	ax.set(xlabel="CCC ")
	figure = ax.get_figure()
	figure.savefig(Path + "/CCC" , dpi=400, bbox_inches='tight')
	plt.show()

saveCCCRadiomic(CCCAllLocValue, FeatursAllLocList)



def saveCCCRadiomicClass(CCC, Featurs, Class):
	df_CustomPalette = pad.DataFrame(CCC, columns = ['CCC'])
	df_CustomPalette['Featurs'] = Featurs
	custom_palette = {}
	for q in set(df_CustomPalette.Featurs):
		val = df_CustomPalette[df_CustomPalette.Featurs == q].CCC
		if val.values < 0.85:
			custom_palette[q] = sns.xkcd_rgb["pale red"]
		else:
			custom_palette[q] = sns.xkcd_rgb["windows blue"]
	plt.figure(figsize=(5,0.36*len(Featurs)))
	plt.axvline(0.85, 0,1, linewidth=3, color='b')
	ax = sns.barplot(x=CCC, y=Featurs, palette=custom_palette)
	ax.set(xlabel="CCC ")
	ax.set_title(Class)
	figure = ax.get_figure()
	figure.savefig(Path + "/CCC_IA" + str(Class), dpi=400, bbox_inches='tight')
	plt.show()



### Discrimination en fonction des classes de radiomics ###

CCCAllLocValueIntensity = CCCAllLocValue[:9]
for elm in CCCAllLocValue[23:41]:
	CCCAllLocValueIntensity.append(elm)
FeatursAllLocListIntensity = FeatursAllLocList[:9]
for elm in FeatursAllLocList[23:41]:
	FeatursAllLocListIntensity.append(elm)

saveCCCRadiomicClass(CCCAllLocValueIntensity, FeatursAllLocListIntensity, "Intensity")






### Double courbe correlation _ fast image VS DL ### IN THE TUMOR ###
df_IA = pad.read_excel("xxx", header=0)
df_input = pad.read_excel("/xxx", header=0)

y=3 #Min
resultsIAAll = pearsonr(df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], df_IA[df_IA.columns[y]][int(len(df_IA)/2):])
resultsInputAll = pearsonr(df_input[df_input.columns[y]][:int(len(df_input)/2)], df_input[df_input.columns[y]][int(len(df_input)/2):])
ax = sns.regplot(x=df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], y=df_IA[df_IA.columns[y]][int(len(df_IA)/2):], data=df_IA)
ax = sns.regplot(x=df_input[df_input.columns[y]][:int(len(df_input)/2)], y=df_input[df_input.columns[y]][int(len(df_input)/2):], data=df_input)
ax.text(0.5, 0.27, "IA IMAGE  (R²=" + str(round(resultsIAAll[0]*resultsIAAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["windows blue"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.text(0.5, 0.19, "FAST IMAGE  (R²=" + str(round(resultsInputAll[0]*resultsInputAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["pale red"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set(xlabel="Original " + str(df_IA.columns[y]) + " values ",  ylabel="Post-processing " + str(df_IA.columns[y])  + " values ")
plt.xlim(120, 400)
plt.ylim(120, 400)
plt.show()
figure = ax.get_figure()
figure.savefig(Path, dpi=400)

#ou
y=8 #Skewness
resultsIAAll = pearsonr(df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], df_IA[df_IA.columns[y]][int(len(df_IA)/2):])
resultsInputAll = pearsonr(df_input[df_input.columns[y]][:int(len(df_input)/2)], df_input[df_input.columns[y]][int(len(df_input)/2):])
ax = sns.regplot(x=df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], y=df_IA[df_IA.columns[y]][int(len(df_IA)/2):], data=df_IA)
ax = sns.regplot(x=df_input[df_input.columns[y]][:int(len(df_input)/2)], y=df_input[df_input.columns[y]][int(len(df_input)/2):], data=df_input, color=sns.xkcd_rgb["pale red"])
ax.text(0.45, 0.14, "FAST IMAGE  (R²=" + str(round(resultsInputAll[0]*resultsInputAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["pale red"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set(xlabel="Original " + str(df_IA.columns[y]) + " values ",  ylabel="Post-processing " + str(df_IA.columns[y])  + " values ")
plt.xlim(-1, 1.5)
plt.ylim(-1, 1.5)
plt.show()
figure = ax.get_figure()
figure.savefig(Path, dpi=400)

y=8 #Skewness
ax = sns.regplot(x=df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], y=df_IA[df_IA.columns[y]][int(len(df_IA)/2):], data=df_IA)
ax.text(0.5, 0.19, "FAST IMAGE  (R²=" + str(round(resultsInputAll[0]*resultsInputAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["pale red"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set(xlabel="Original " + str(df_IA.columns[y]) + " values ",  ylabel="Post-processing " + str(df_IA.columns[y])  + " values ")
plt.xlim(-1, 1.5)
plt.ylim(-1, 1.5)
plt.show()
figure = ax.get_figure()
figure.savefig(Path, dpi=400)



#ou
y=9 # kurtosis
resultsIAAll = pearsonr(df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], df_IA[df_IA.columns[y]][int(len(df_IA)/2):])
resultsInputAll = pearsonr(df_input[df_input.columns[y]][:int(len(df_input)/2)], df_input[df_input.columns[y]][int(len(df_input)/2):])
ax = sns.regplot(x=df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], y=df_IA[df_IA.columns[y]][int(len(df_IA)/2):], data=df_IA)
ax = sns.regplot(x=df_input[df_input.columns[y]][:int(len(df_input)/2)], y=df_input[df_input.columns[y]][int(len(df_input)/2):], data=df_input)
ax.text(0.08, 0.80, "IA IMAGE  (R²=" + str(round(resultsIAAll[0]*resultsIAAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["windows blue"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.text(0.08, 0.72, "FAST IMAGE  (R²=" + str(round(resultsInputAll[0]*resultsInputAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["pale red"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set(xlabel="Reference image " + str(df_IA.columns[y]) + " values ",  ylabel="Post-processing " + str(df_IA.columns[y])  + " values ")
plt.xlim(0, 2.5)
plt.ylim(0, 2.5)
plt.show()
figure = ax.get_figure()
figure.savefig(Path, dpi=400)

#ou
y=49 # glcm MCC (complexité)
resultsIAAll = pearsonr(df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], df_IA[df_IA.columns[y]][int(len(df_IA)/2):])
resultsInputAll = pearsonr(df_input[df_input.columns[y]][:int(len(df_input)/2)], df_input[df_input.columns[y]][int(len(df_input)/2):])
ax = sns.regplot(x=df_IA[df_IA.columns[y]][:int(len(df_IA)/2)], y=df_IA[df_IA.columns[y]][int(len(df_IA)/2):], data=df_IA)
ax = sns.regplot(x=df_input[df_input.columns[y]][:int(len(df_input)/2)], y=df_input[df_input.columns[y]][int(len(df_input)/2):], data=df_input)
ax.text(0.5, 0.27, "IA IMAGE  (R²=" + str(round(resultsIAAll[0]*resultsIAAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["windows blue"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.text(0.5, 0.19, "FAST IMAGE  (R²=" + str(round(resultsInputAll[0]*resultsInputAll[0],3)) +")", weight="bold", fontsize=12, color=sns.xkcd_rgb["pale red"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
ax.set(xlabel="Original " + str(df_IA.columns[y]) + " values ",  ylabel="Post-processing " + str(df_IA.columns[y])  + " values ")
plt.xlim(0.55, 0.95)
plt.ylim(0.55, 0.95)
plt.show()
figure = ax.get_figure()
figure.savefig(Path, dpi=400)



def normalityDistributionTestAgostino(df):

	### avec test D’Agostino
	y=3 
	for i in range(118): 
		NormOrigin = stats.normaltest(df[df.columns[y]][:int(len(df)/2)])
		NormIA = stats.normaltest(df[df.columns[y]][int(len(df)/2):])
		if NormOrigin[1]> 0.05:
			print("La variable " + str(df.columns[y] + " pour les données Origin, ne suit pas une loi normale (p = " + str(NormOrigin[1])) + ")")
		if NormIA[1]> 0.05:
			print("La variable " + str(df.columns[y] + " pour les données IA, ne suit pas une loi normale (p = " + str(NormIA[1])) + ")")
		y+=1
	### avec test D’Agostino

normalityDistributionTestAgostino(df)




### Radiomic model predictive values differences
df = pad.read_excel("xxx", header=0)
Path = 'xxx'

print(stats.ttest_rel(df['model_resampling_su'][0:40], df['model_resampling_su'][40:80]))
print(stats.ttest_rel(df['model_resampling_su'][0:40], df['model_resampling_su'][80:]))

print(stats.ttest_rel(df['model_resampling_chien'][0:40], df['model_resampling_chien'][40:80]))
print(stats.ttest_rel(df['model_resampling_chien'][0:40], df['model_resampling_chien'][80:]))

print(stats.ttest_rel(df['model_denoising_su'][0:40], df['model_denoising_su'][40:80]))
print(stats.ttest_rel(df['model_denoising_su'][0:40], df['model_denoising_su'][80:]))

print(stats.ttest_rel(df['model_denoising_chien'][0:40], df['model_denoising_chien'][40:80]))
print(stats.ttest_rel(df['model_denoising_chien'][0:40], df['model_denoising_chien'][80:]))