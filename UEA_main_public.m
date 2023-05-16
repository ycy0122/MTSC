clc;clear;close all
data_multivariate = '../Multivariate_arff';

addpath(genpath(data_multivariate))
data_listing = dir(data_multivariate);

file_num = size(data_listing,1);
multi_listing_cell = struct2cell(data_listing).';
dir_id = cell2mat(multi_listing_cell(:,5));
file_type = multi_listing_cell(dir_id,1);
file_type = file_type(3:end);
file_type = file_type([1:3,5:15,19:28, 30, 31]);

CV = 5;

Mtemrs_Sp = [6,2,5,5,5,    5,3,3,5,10,...
            10,10,3,10,2,  5,5,10,2,10,...
            10,10,10,7,4,  3];
Mtemrs_Tp = [6,20,5,5,10,  5,10,5,10,10,...
            25,5,5,5,10,   5,10,5,10,5,...
            10,5,5,10,5,   20];
Dilation_list = [1,1,5,2,5, 1,3,1,5,1,...
                 2,1,1,5,1, 1,1,3,1,1,...
                 1,3,2,2,5, 2];
Window_length = [48,640,48,191 90, 65,562,25,50,31,...
               50,100,50,101,36, 22,1000,17,72,8,...
               50,10,896,576,450, 315];

for i = [1:26]
    clear M1_leg_tr M1_leg_te
    Mtemrs_Spatial = Mtemrs_Sp(i);
    Mterms_temporal = Mtemrs_Tp(i);
    dilation = Dilation_list(i);

    folder_name = file_type{i};
    datafolder = [data_multivariate,'/',folder_name];
    current_folder = pwd;
    cd(datafolder)
    arffFiles = dir([folder_name,'*.arff']);
    numfiles = size(arffFiles,1);

    traindata = [data_multivariate,'/',folder_name,'/',folder_name,'_TRAIN.arff'];
    testdata =  [data_multivariate,'/',folder_name,'/',folder_name,'_TEST.arff'];
    
    [X_train,X_test,Y_train,Y_test,N_train,N_test] = UEA_readdata(folder_name,arffFiles,numfiles);
    % classification for consecutive window
    Y = [Y_train; Y_test];
    c = cvpartition(Y,'KFold',CV);

    L = Window_length(i);
    window_number = floor(size(X_train,2)/Window_length(i));
    % features used
    LEG2_DIL_tr = [];
    LEG2_DIL_te = [];
    LEG_MT_SFT_tr = [];
    LEG_MT_SFT_te = [];
    SIG_X_tr = [];
    SIG_X_te = [];
    KPM_X_tr = [];
    KPM_X_te = [];

    for w = 1:window_number

        New_MPt2_tr = [];
        New_MPt2_te = [];

        M2_cell_leg2_tr = {};
        M2_dil_leg2_te = {};
        X_dyn_tr = cell(1,dilation);
        Xkp_dyn_tr = cell(1,dilation);
        X_dyn_te = cell(1,dilation);
        Xkp_dyn_te = cell(1,dilation);

        skip_tr = [];
        skip_te = [];
       
        Xkp_dyn_tr_new = [];
        Xkp_dyn_te_new = [];

        X_dyn_tr_new = [];
        M2_dil_tr_new = [];
        X_dyn_te_new = [];
        M2_dil_te_new = [];


        this_window_train = X_train(:,(w-1)*Window_length(i)+1 : (w)*Window_length(i),:);
        this_window_test = X_test(:,(w-1)*Window_length(i)+1 : (w)*Window_length(i),:);
    
        a = -1; b = 1;               % lowerbound and upper bound for beta
        beta = linspace(a,b,size(this_window_train,1)); 
    
        P = zeros(Mtemrs_Spatial, size(this_window_train,1));
        for jj = 0:1:Mtemrs_Spatial-1
            P(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta);
        end
    
        Pt = zeros(Mterms_temporal, size(this_window_train,2));
        beta2 = linspace(a,b,size(this_window_train,2)); 
        for jj = 0:1:Mterms_temporal-1
            Pt(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta2);
        end
    
        Pt_dil = cell(1, dilation);
        for di = 1:dilation
            beta2_new = beta2(1:di:end);
            for jj = 0:1:Mterms_temporal-1
               Pt_dil{di}(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta2_new);
            end
        end

        % spectral Moment
        for j = 1:size(this_window_train,3)   % cases
            t = linspace(0,1,size(this_window_train,2));
            [New_MPt2_tr(:,:,j)] = CalTempLegMomentsbeta2, this_window_train(:,:,j), Mterms_temporal);
    
            for di = 1:dilation
                this_X = this_window_train(:,:,j);
                mask = zeros(size(this_X));
                r = 1:di:size(this_X,2);
                mask(:,r) = 1;
                this_X = this_X.*logical(mask);
                for k = 1:Mterms_temporal   % Legendre temporal moments
                    M2_cell_leg2_tr{di}(k,:,j) = (P.'\this_X)*(Pt(k,:).');
                end
            end

    %         generate Koopman features
            for k = 1:dilation
                this_X = this_window_train(:,:,j);
    
                mask = zeros(size(this_X));
                r = 1:k:size(this_X,2);
                mask(:,r) = 1;
                this_X = this_X.*logical(mask);
    
                X1 = this_X(:,1:end-1);
                X2 = this_X(:,2:end);
                [U, S, V] = svd(X1 , 'econ');
                DMD_terms_taken = min(Mtemrs_Spatial, size(U,2));
      
                if size(M1_leg_tr(:,1:k:end,:),2) >= size(M1_leg_tr,1)
                    [CX_Kp,~,~] = genFeature_MomentKp(this_X, floor(size(this_X,2)/(2))  , 'Prony');
                elseif size(M1_leg_tr(:,1:k:end,:),2) < size(M1_leg_tr,1)
                    [CX_Kp,~,~] = genFeature_MomentKp(this_X, floor(size(this_X,2))-1  , 'Arnoldi');
                end 
                    Xkp_dyn_tr{k}(:,:,j) = CX_Kp;
                    X_dyn_tr{k}(:,:,j) = diag(S); %CX_Kp;
            end
        end
    
        a = -1; b = 1;               % lowerbound and upper bound for beta
        beta = linspace(a,b,size(this_window_test,1)); 
    
        P = zeros(Mtemrs_Spatial, size(this_window_test,1));
        for jj = 0:1:Mtemrs_Spatial-1
            P(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta);
        end
    
        Pt = zeros(Mterms_temporal, size(this_window_test,2));
        beta2 = linspace(a,b,size(this_window_test,2)); 
        for jj = 0:1:Mterms_temporal-1
            Pt(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta2);
        end
    
        Pt_dil = cell(1, dilation);
        for di = 1:dilation
            beta2_new = beta2(1:di:end);
            for jj = 0:1:Mterms_temporal-1
               Pt_dil{di}(jj+1,:) = sqrt((2*(jj)+1)./2).*legpoly(jj,beta2_new);
            end
        end

        % spectral Moment
        for j = 1:size(this_window_test,3)   % cases
            [New_MPt2_te(:,:,j)] = CalTempLegMoments(beta2, this_window_test(:,:,j), Mterms_temporal);
    
            for di = 1:dilation
                for k = 1:Mterms_temporal   % Legendre temporal moments
                    M2_dil_leg2_te{di}(k,:,j) = (P.'\this_window_test(:,1:di:end,j))*(Pt_dil{di}(k,:).');
                end
            end
    %         generate Koopman features
            for k = 1:dilation
                this_X = this_window_test(:,:,j);
                
                mask = zeros(size(this_X));
                r = 1:k:size(this_X,2);
                mask(:,r) = 1;
                this_X = this_X.*logical(mask);
    
                X1 = this_X(:,1:end-1);
                X2 = this_X(:,2:end);
                [U, S, V] = svd(X1 , 'econ');
                DMD_terms_taken = min(Mtemrs_Spatial, size(U,2));
     
                if size(M1_leg_te(:,1:k:end,:),2) >= size(M1_leg_te,1)
                    [CX_Kp,~,~] = genFeature_MomentKp(this_X, floor(size(this_X,2)/(2))  , 'Prony');
                elseif size(M1_leg_te(:,1:k:end,:),2) < size(M1_leg_te,1)
                    [CX_Kp,~,~] = genFeature_MomentKp(this_X, floor(size(this_X,2))-1  , 'Arnoldi');
                end 
                    Xkp_dyn_te{k}(:,:,j) = CX_Kp;
                    X_dyn_te{k}(:,:,j) = diag(S); %CX_Kp;
            end
        end

        for k = 1:dilation
            X_dyn_tr_new = [X_dyn_tr_new;X_dyn_tr{k}(:,:,setdiff(1:size(X_dyn_tr{k},3),skip_tr))];
            Xkp_dyn_tr_new = [Xkp_dyn_tr_new;Xkp_dyn_tr{k}(:,:,setdiff(1:size(Xkp_dyn_tr{k},3),skip_tr))];
            M2_dil_tr_new = [M2_dil_tr_new; M2_cell_leg2_tr{k}(:,:,setdiff(1:size(X_dyn_tr{k},3),skip_tr))];
        
            X_dyn_te_new = [X_dyn_te_new;X_dyn_te{k}(:,:,setdiff(1:size(X_dyn_te{k},3),skip_te))];
            Xkp_dyn_te_new = [Xkp_dyn_te_new;Xkp_dyn_te{k}(:,:,setdiff(1:size(Xkp_dyn_te{k},3),skip_te))];
            M2_dil_te_new = [M2_dil_te_new; M2_dil_leg2_te{k}(:,:,setdiff(1:size(X_dyn_te{k},3),skip_te))];
        end

        % tidy things up ...    
        % dynamics - enture sequence
        X_dyn_tr = reshape(X_dyn_tr_new,[],N_train-length(skip_tr));
        X_dyn_te = reshape(X_dyn_te_new,[],N_test-length(skip_te));
        Xkp_dyn_tr = reshape(Xkp_dyn_tr_new ,[],N_train-length(skip_tr));
        Xkp_dyn_te = reshape(Xkp_dyn_te_new ,[],N_test-length(skip_te));
    
        % spatial temporal moments - entire sequence
        M2_dil_tr_new = reshape(M2_dil_tr_new, [], N_train-length(skip_tr));
        M2_dil_te_new = reshape(M2_dil_te_new, [], N_test-length(skip_te));
    
        % temporal moments - new
        New_MPt2_te = reshape(New_MPt2_te(:,[1,3:end],:), [], N_test-length(skip_te));
        New_MPt2_tr = reshape(New_MPt2_tr(:,[1,3:end],:), [], N_train-length(skip_tr));
    
        % for those features actually used...

        LEG2_DIL_tr = [LEG2_DIL_tr;M2_dil_tr_new];
        LEG2_DIL_te = [LEG2_DIL_te;M2_dil_te_new];

        LEG_MT_SFT_tr = [LEG_MT_SFT_tr; New_MPt2_tr];
        LEG_MT_SFT_te = [LEG_MT_SFT_te; New_MPt2_te];

        SIG_X_tr = [SIG_X_tr; X_dyn_tr];
        SIG_X_te = [SIG_X_te; X_dyn_te];

        KPM_X_tr = [KPM_X_tr; Xkp_dyn_tr];
        KPM_X_te = [KPM_X_te; Xkp_dyn_te];

        % Feature by window... may not be good
        wLEG2_DIL_tr = M2_dil_tr_new;
        wLEG2_DIL_te = M2_dil_te_new;

        wLEG_MT_SFT_tr = New_MPt2_tr;
        wLEG_MT_SFT_te = New_MPt2_te;

        wSIG_X_tr = X_dyn_tr;
        wSIG_X_te = X_dyn_te;

        wKPM_X_tr = Xkp_dyn_tr;
        wKPM_X_te = Xkp_dyn_te;


        % consecutive window
        SpTpDyn2 = [[wLEG_MT_SFT_tr;wSIG_X_tr;wKPM_X_tr; wLEG2_DIL_tr], [wLEG_MT_SFT_te; wSIG_X_te; wKPM_X_te; wLEG2_DIL_te]];

        tree = templateTree('PredictorSelection','interaction-curvature','Surrogate','on',...
            'SplitCriterion','deviance'); 
        for ii = 1:CV        
            Mdl_SpTpDyn2 = fitcensemble(SpTpDyn2(:,c.training(ii)).', Y(c.training(ii)),'Method','Bag', ...
            'NumLearningCycles',200,'Learners',tree);
                label_SpTpDyn2{:,w,ii} = predict(Mdl_SpTpDyn2,SpTpDyn2(:,c.test(ii)).');
                accuracy_SpTpDyn2(ii,1) = 100.*sum(label_SpTpDyn2{:,w,ii}==Y(c.test(ii)))./length(Y(c.test(ii)));
        
        end

    end

    % 
    SpTpDyn2 = [[LEG_MT_SFT_tr;SIG_X_tr;KPM_X_tr; LEG2_DIL_tr], [LEG_MT_SFT_te;SIG_X_te;KPM_X_te; LEG2_DIL_te]];

%% one-NN
clc
tree = templateTree('PredictorSelection','interaction-curvature','Surrogate','on',...
    'SplitCriterion','deviance'); 
for ii = 1:CV
    Mdl_SpTpDyn2 = fitcensemble(SpTpDyn2(:,c.training(ii)).', Y(c.training(ii)),'Method','Bag', ...
    'NumLearningCycles',200,'Learners',tree);
        label = predict(Mdl_SpTpDyn2,SpTpDyn2(:,c.test(ii)).');
        vlabel_SpTpDyn2 = mode([label_SpTpDyn2{:,:,ii},label],2);
        accuracy_SpTpDyn2(ii,1) = 100.*sum(vlabel_SpTpDyn2==Y(c.test(ii)))./length(Y(c.test(ii)));
end

end
