class CFG:
    # model architecture hyper-parameters
    emb_len = 200
    code_len = 250
    max_seq_len = 20
    mask_token_ratio = 0.2
    # query embedder
    ignore_query = True
    query_embedder_nhead = 4
    query_embedder_nhid = 800
    query_embedder_nlayers = 4
    query_embedder_dropout = 0.
    query_max_len = 20
    # gnn
    gnn_nhid = 800
    gnn_nout = 800#1000
    gnn_nhead = 4
    gnn_aggr = 'sum'
    gnn_dropout = 0.
    gnn_pooling_out = 400
    gnn_nlayers = 3
    # product head
    ph_nhid = 400
    ph_nlayers = 1
    ph_dropout = 0.
    # query head
    qh_nhead = 5
    qh_nhid = 768
    qh_nlayers = 1
    qh_dropout = 0.
    # sim loss
    #sim_w = 0.
    #sim_type = 'STAN'
    #STAN_lammy = 1.04
    # emb
    n_out = 500
    # training hyper-parameters
    node_mask_prob = 0.05
    batch_size = 50
    ft_batch_size = 10
    lr = 0.0003
    weight_decay = 0.0000
    ph_w = 0.
    qh_w = 0.
    pt_w = 0.
    ctv_w = 0.
    bin_w = 0.3
    qaea_w = 0.
    node_w = 0.
    token_w = 0.
    max_epoch = 60
    neg_k = 10
    rec_w = 1.
    aux_w = 20
    max_train_num = 1000000
    ckpt_iter = 500
    mask_prob = 0.
    #fine tune
    fine_tune_data_num = 10000
    loss_type = 'MSE'
    sim_type = 'all_product_type_score'
    load_path = 'SavedModel/FilteredAmazon-CL-02-next-all-HGGNN-SrGNNPooling-3/pretrain_model.pth'
    fine_tune_epoch = 70
    # tokenizer
    token_len = 20
    # save dir
    #savedir = "SavedModel/Yoochoose-next-all-HGGNN-SrGNNPooling-2/"
    savedir = 'SavedModel/gce-gnn-plus/'
    #savedir = "SavedModel/FilteredAmazon-CL-02-next-all-HGGNN-SrGNNPooling-pt-pid-fq-ctv-node-token/"
    #savedir = "SavedModel/FilteredAmazon"
    #savedir = "SavedModel/FilteredAmazon-QAEA-FT/"
    log_file = savedir+"train.log"