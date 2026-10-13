"""Held-out regression metrics and complementary model-effect plots."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from download_data import download
ROOT = Path(__file__).parent

def ale_1d(model, X, feature, bins=15):
    """Quantile-bin finite differences, accumulated and observation-weight centered."""
    values = X[feature].to_numpy()
    edges = np.unique(np.quantile(values,np.linspace(0,1,bins+1)))
    if len(edges)<2:
        raise ValueError('ALE requires a nonconstant feature')
    ids = np.clip(np.searchsorted(edges,values,side='right')-1,0,len(edges)-2)
    differences=[];counts=[]
    for i in range(len(edges)-1):
        group=X.loc[ids==i].copy();counts.append(len(group))
        if group.empty:
            differences.append(0.);continue
        low=group.copy();high=group.copy()
        low[feature]=edges[i];high[feature]=edges[i+1]
        differences.append(float(np.mean(model.predict(high)-model.predict(low))))
    accumulated=np.cumsum(differences)-np.asarray(differences)/2
    accumulated-=np.average(accumulated,weights=counts)
    return (edges[:-1]+edges[1:])/2, accumulated

def run():
    path=download();data=pd.read_csv(path)
    X=data.drop(columns='median_house_value');y=data['median_house_value']
    train,test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    numeric=[c for c in X if c!='ocean_proximity']
    def prep():
        return ColumnTransformer([('numeric',SimpleImputer(strategy='median'),numeric),('coast',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['ocean_proximity'])])
    models={'Mean baseline':make_pipeline(prep(),DummyRegressor()),'Linear Regression':make_pipeline(prep(),LinearRegression()),
            'Random Forest':make_pipeline(prep(),RandomForestRegressor(n_estimators=150,max_features=.8,random_state=42,n_jobs=2))}
    out=ROOT/'reports';out.mkdir(exist_ok=True);rows=[]
    for name,model in models.items():
        model.fit(train,y_train);p=model.predict(test)
        rows.append({'Model':name,'MAE':mean_absolute_error(y_test,p),'RMSE':np.sqrt(mean_squared_error(y_test,p)),'R2':r2_score(y_test,p)})
    metrics=pd.DataFrame(rows).set_index('Model');metrics.to_csv(out/'metrics.csv')
    forest=models['Random Forest'];sample=test.sample(min(1200,len(test)),random_state=42)
    importance=permutation_importance(forest,sample,y_test.loc[sample.index],n_repeats=5,scoring='neg_mean_absolute_error',random_state=42,n_jobs=1)
    imp=pd.DataFrame({'Feature':X.columns,'MAE increase':importance.importances_mean,'Std':importance.importances_std}).sort_values('MAE increase')
    imp.to_csv(out/'permutation_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5));ax.barh(imp['Feature'],imp['MAE increase'],xerr=imp['Std'],color='#244b69');ax.set_xlabel('Increase in MAE after shuffling ($)');ax.set_title('Which inputs matter to held-out predictions?');fig.tight_layout();fig.savefig(out/'permutation_importance.png',dpi=160);plt.close(fig)
    background=train.sample(600,random_state=42)
    fig,ax=plt.subplots(figsize=(8,5));display=PartialDependenceDisplay.from_estimator(forest,background,['median_income'],kind='both',subsample=35,grid_resolution=25,random_state=42,ax=ax)
    effect_ax=display.axes_[0,0]
    effect_ax.set_xlabel('Median income (tens of thousands of 1990 dollars)');effect_ax.set_ylabel('Predicted median value ($)');ax.set_title('PDP / ICE: average association and variation');fig.tight_layout();fig.savefig(out/'pdp_ice.png',dpi=160);plt.close(fig)
    grid,effect=ale_1d(forest,train.sample(3000,random_state=42),'median_income')
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(grid,effect,marker='o',color='#244b69');ax.axhline(0,color='#aaa',ls='--');ax.set_xlabel('Median income (tens of thousands of 1990 dollars)');ax.set_ylabel('Centered model effect ($)');ax.set_title('ALE: local differences within observed income bins');fig.tight_layout();fig.savefig(out/'ale.png',dpi=160);plt.close(fig)
    (out/'run.json').write_text(json.dumps({'seed':42,'split':'random 80/20','rows':len(data),'train_rows':len(train),'test_rows':len(test),
       'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'preprocessing_fit':'training only','validation_scope':'random split; no spatial holdout or hyperparameter search',
       'permutation_rows':len(sample),'ALE_bins':15},indent=2)+'\n')
    print(metrics.round(4).to_string());return metrics
if __name__=='__main__':run()
