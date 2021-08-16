# LearningToRank

# LTR_tfr.ipynb
tensorflow-ranking is a library which helps in building Learning to Rank (LTR) models which use neural networks. This .ipynb notebook is an attempt to explain how to use this library to rank
documents given various (Query,set(documents)) pairs.

The notebook consists of the following parts:

1. Data Simulation

   This part simulates data where there are total 4000 documents and each query has 4 documents mapped to itself. So total 1000 queries. There are 5 features and the target label is 'Relevancy' which is binary in nature.

2. Training LTR model

  In this part a pointwise LTR model is built(Group size=1) and all the implementation is taken from https://github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/tf_ranking_libsvm.py. While training, the model is saved in a predefined directory.
  
3. Dummy model creation

  One of the challenges faced while using this library was making predictions using the trained model. The trained model which is wrapped in an estimator when used directly (estimator.predict(...)) did not work. So as a workaround, a dummy estimator is created and then the saved checkpoints are used to load the trained saved model.
  
4. SHAP

  SHAP.KernelExplainer is used to explain the predictions. 
  
  The predictions are made as follows using the estimator:
  
    predictions=list(est.predict(test_input_fn,hooks=[test_hook]))

  However, test_input_fn,hooks=[test_hook]-> invalid to pass to SHAP because while building an explainer in SHAP the following is done:
  
    explainer = shap.KernelExplainer(f, x)
    
    Where x is numpy or df and is passed as parameter to function f.
  
  To connect SHAP with the estimator the following connecting function is written:
  
      def makepred(df_new):
        features=load_libsvm_datas(df_new,4)
        test_input_fn, test_hook = get_eval_inputss(features)
        pred=est.predict(test_input_fn,hooks=[test_hook])
        gt=np.array(list(pred))
        ypred=[]
        for g in gt:
          for f in (g):
            ypred.append(f)
        return np.array(ypred)

    explainer=shap.KernelExplainer(makepred,df.iloc[0:2000][['F1','F2','F3','F4','F5']])

  
