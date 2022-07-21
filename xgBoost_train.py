from root_numpy import root2array, rec2array, tree2array
from ROOT import TChain
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc
#from sklearn.externals import joblib
import joblib
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from root_pandas import to_root, read_root
import os, sys
import seaborn as sn
import root_numpy
import xgboost as xgb
from numpy import loadtxt
from sklearn.metrics import accuracy_score

def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=60):
    decisions = []
    for X,y in ((X_train, y_train), (X_test, y_test)):
#        d1 = clf.decision_function(X[y>0.5]).ravel()
#        d2 = clf.decision_function(X[y<0.5]).ravel()
        d1 = clf.predict_proba(X[y > 0.5])[:,1].ravel()
        d2 = clf.predict_proba(X[y < 0.5])[:,1].ravel()

        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    #low_high = (low,high)
    #low_high = (-20,15)
    low_high = (0.0,1.0)

    fig, ax = plt.subplots()
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    fig.savefig("figs/Train_Test_BDT_Bu2Dh_PIDD_exc.pdf")

    plt.yscale('log')
    fig.savefig("figs/Train_Test_BDT_Bu2Dh_log_PID_exc.pdf")

#Training variables
branch_names = ["B_ptasy_1.50",
			    "log(B_LoKi_FDCHI2_BPV)",
			    "log(D0_LoKi_FDCHI2_BPV)",
			    "Bach_P",
			    "Bach_PT",
			    "D0_PT",
			    "log(Bach_IPCHI2_OWNPV)",
			    "log10(B_LoKi_MIPCHI2_PV)",
			    "log(D0_LoKi_MIPCHI2_PV)",
			    "B_LoKi_MAXDOCA",
			    "D0_LoKi_AMAXDOCA",
			    "log(1-B_LoKi_DIRA_BPV)",
			    "log(1-D0_LoKi_DIRA_BPV)",
			    "log10(B_LoKi_IP_BPV)",
			    "log10(D0_LoKi_IP_BPV)",
			    "log10(B_LoKi_RHO_BPV)",
			    "log10(B_LoKi_FD_BPV)",
			    "log10(D0_LoKi_RHO_BPV)",
			    "K_PIDK",
			    "Pi_PIDK"
				]

branch_names_mc = ["B_ptasy_1.50",
                            "log(B_LoKi_FDCHI2_BPV)",
                            "log(D0_LoKi_FDCHI2_BPV)",
                            "Bach_P",
                            "Bach_PT",
                            "D0_PT",
                            "log(Bach_IPCHI2_OWNPV)",
                            "log10(B_LoKi_MIPCHI2_PV)",
                            "log(D0_LoKi_MIPCHI2_PV)",
                            "B_LoKi_MAXDOCA",
                            "D0_LoKi_AMAXDOCA",
                            "log(1-B_LoKi_DIRA_BPV)",
                            "log(1-D0_LoKi_DIRA_BPV)",
                            "log10(B_LoKi_IP_BPV)",
                            "log10(D0_LoKi_IP_BPV)",
                            "log10(B_LoKi_RHO_BPV)",
                            "log10(B_LoKi_FD_BPV)",
                            "log10(D0_LoKi_RHO_BPV)",
                            "K_PIDK_corr",
                            "Pi_PIDK_corr"
                                ]

branch_names_nolog = ["B_ptasy_1.50",
			    "B_LoKi_FDCHI2_BPV",
			    "D0_LoKi_FDCHI2_BPV",
			    "Bach_P",
			    "Bach_PT",
			    "D0_PT",
			    "Bach_IPCHI2_OWNPV",
			    "B_LoKi_MIPCHI2_PV",
			    "D0_LoKi_MIPCHI2_PV",
			    "B_LoKi_MAXDOCA",
			    "D0_LoKi_AMAXDOCA",
			    "B_LoKi_DIRA_BPV",
			    "D0_LoKi_DIRA_BPV",
			    "B_LoKi_IP_BPV",
			    "D0_LoKi_IP_BPV",
			    "B_LoKi_RHO_BPV",
			    "B_LoKi_FD_BPV",
			    "D0_LoKi_RHO_BPV",
                            "K_PIDK",
                            "Pi_PIDK"
				]

branch_names_nolog_mc = ["B_ptasy_1.50",
                            "B_LoKi_FDCHI2_BPV",
                            "D0_LoKi_FDCHI2_BPV",
                            "Bach_P",
                            "Bach_PT",
                            "D0_PT",
                            "Bach_IPCHI2_OWNPV",
                            "B_LoKi_MIPCHI2_PV",
                            "D0_LoKi_MIPCHI2_PV",
                            "B_LoKi_MAXDOCA",
                            "D0_LoKi_AMAXDOCA",
                            "B_LoKi_DIRA_BPV",
                            "D0_LoKi_DIRA_BPV",
                            "B_LoKi_IP_BPV",
                            "D0_LoKi_IP_BPV",
                            "B_LoKi_RHO_BPV",
                            "B_LoKi_FD_BPV",
                            "D0_LoKi_RHO_BPV",
                            "K_PIDK_corr",
                            "Pi_PIDK_corr"
                                ]

path = "/data/lhcb/users/abrantes/Bc2DX_crosscheck/ntuples"

MC_years = ["2011","2012","2015","2016","2017","2018"]
MC_mags = ["Up","Down"]

sigtree = TChain("DecayTree")

for year in MC_years:
	for mag in MC_mags:
		sigtree.Add(path+"/MC/Bc2DK/%s_%s/Total_Bc2DK_Cuts_PIDCorr.root" % (year,mag))
		print(path+"/MC/Bc2DK/%s_%s/Total_Bc2DK_Cuts_PIDCorr.root" % (year,mag))

print("Number of MC signal events: %s" % sigtree.GetEntries())
print("Number of MC signal events used: %s" % sigtree.GetEntries("B_BKGCAT==20"))

sig = tree2array(sigtree,
    branch_names_mc,
    selection='B_BKGCAT==20'
    )
sig = rec2array(sig)

print("Signal array created")

bkgtree = TChain("DecayTree")

data_years = ["2011","2012","2015","2016","2017","2018"]
data_mags = ["Up","Down"]

for year in data_years:
	for mag in data_mags:
		bkgtree.Add(path+"/data/Bu2Dh/%s_%s/Total_Bu2DK_Cuts.root" % (year,mag))
		print(path+"/data/Bu2Dh/%s_%s/Total_Bu2DK_Cuts.root" % (year,mag))

bkgtree.SetBranchStatus("*",0)
for var in branch_names_nolog:
	bkgtree.SetBranchStatus(var,1)
bkgtree.SetBranchStatus("B_D0constPVconst_M",1)

print("Number of data events: %s" % bkgtree.GetEntries())
print("Number of data events used: %s" % bkgtree.GetEntries("B_D0constPVconst_M>5900 && abs(B_D0constPVconst_M-6274.9)>(20.8*3.0)"))

bkg = tree2array(bkgtree,
    branch_names,
    selection='B_D0constPVconst_M>5900 && abs(B_D0constPVconst_M-6274.9)>(20.8*3.0)', #Tain on events in the upper mass sideband
    step = 100 #Select random 1% of the background sample to use
    )
bkg = rec2array(bkg)

print("Background array created")

#Correlation Matrix Signal and Background
print("Plotting correlation matrices...")

plt.rcParams['figure.figsize'] = [12, 12] # for square canvas
fig, ax = plt.subplots()
df_sig = pd.DataFrame(sig,columns = branch_names)
corrMatrix_sig = df_sig.corr()
sn.heatmap(corrMatrix_sig, annot=True, fmt='.1g')
plt.title('Signal Correlation Matrix')
plt.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=8)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
#plt.show()
fig.savefig("figs/CorrMatrix_sig_PIDD_exc.pdf")

fig, ax = plt.subplots()
df_bkg = pd.DataFrame(bkg,columns = branch_names)
corrMatrix_bkg = df_bkg.corr()
sn.heatmap(corrMatrix_bkg, annot=True, fmt='.1g')
plt.title('Background Correlation Matrix')
plt.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=8)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
#plt.show()
fig.savefig("figs/CorrMatrix_bkg_PIDD_exc.pdf")

print("Done!")
plt.rcParams['figure.figsize'] = [10, 10] # for square canvas
ax.tick_params(axis='both', which='major', labelsize=10)

#Combine into single dataset for use in scikit
X = np.concatenate((sig,bkg))
y = np.concatenate((np.ones(sig.shape[0]),
                    np.zeros(bkg.shape[0])))

X_train,X_test, y_train,y_test = train_test_split(X, y,
                                                  test_size=0.25, random_state=42)

#Account for different stats in the signal and background samples
weights = compute_sample_weight(class_weight='balanced', y=y_train)

#bdt = GradientBoostingClassifier(n_estimators=1000, max_depth=1, learning_rate=0.1, min_samples_split=2,verbose=1)
#TMVA defaults
#bdt = GradientBoostingClassifier(n_estimators=1000, max_depth=2, learning_rate=0.1, min_samples_split=2,verbose=1)

#xgboost
bdt = xgb.XGBClassifier(n_estimators=800, max_depth=2, learning_rate=0.1, verbosity=1, tree_method='exact',use_label_encoder=False,eval_metric='auc')
print(bdt)

print("Fitting model...")

bdt.fit(X_train, y_train,sample_weight=weights)

#Get the feature importances
feature_importances = pd.DataFrame(bdt.feature_importances_,
                                   index = branch_names,
                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)
bdt_name = "xgboost"

#Assess the performance on a sample not used in the training
y_predicted = bdt.predict(X_test)
print(classification_report(y_test, y_predicted,target_names=["background", "signal"]))
#print("Area under ROC curve: %.4f"%(roc_auc_score(y_test,bdt.decision_function(X_test))))
print(":: {} bdt train :: area under ROC curve :: {:.4f} ::".format(bdt_name, roc_auc_score(y_test, bdt.predict_proba(X_test)[:,1])))

#Create ROC curve
fig, ax = plt.subplots()
#decisions = bdt.decision_function(X_test)
decisions = bdt.predict_proba(X_test)[:,1]
# Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(y_test, decisions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Background efficiency')
plt.ylabel('Signal efficiency')
plt.title('ROC')
plt.legend(loc="lower right")
plt.grid()
fig.savefig("figs/ROC_BDT_Bu2Dh_PIDD_exc.pdf")
#plt.show()

#Overtraining test
compare_train_test(bdt, X_train, y_train, X_test, y_test)

#Save the classifier (for later use in applying weights to the samples)
if os.path.exists('output/bdt.joblib'):
	os.remove('output/bdt.joblib')
joblib.dump(bdt, 'output/bdt.joblib')
#Can later be read back in with this
#bdt = joblib.load('output/bdt.joblib')
