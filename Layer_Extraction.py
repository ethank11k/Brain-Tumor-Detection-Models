model_feat_res = Model(inputs=input_img,outputs=final_model.get_layer('dense_4').output)
feat_train_res = model_feat_res.predict(X_train_prep)
feat_val_res = model_feat_res.predict(X_val_prep)
feat_test_res = model_feat_res.predict(X_test_prep)


