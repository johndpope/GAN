# LANGUAGE-STYLE-TRANSFER-FROM-NON-PARALLEL-TEXT-WITH-ARBITRARY-STYLES-Pytorch-Implement
This Project is the Implement of the Paper LANGUAGE STYLE TRANSFER FROM NON-PARALLEL TEXT WITH ARBITRARY STYLES with pytorch, and mainly focus on Chinese.
## src:
# ModelDefine.py
> don't use!

# PretrainDs.py
> python PretrainDs.py "style_data_filename" "train_data_filename" "if_build_new_Model" "Ds_model_name" "embedding_model_name" "epoches"

# trianVAE_D.py
> python --epoches=N --batch_size=N --pretrainD/--PretrainVAE --traindata=path --style=path --gan=path --ds=path --ds_emb=path

# buildModel.py
> python --style=path --output=path
