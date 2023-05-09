function [X_train,X_test,Y_train,Y_test,N_train,N_test] = UCI_readdata(folder_name,arffFiles,numfiles)
    % this file reads multivariate time series dataste and process for ML
    % classification tasks
    %
    if strcmp(folder_name,'ArticularyWordRecognition')
        m = 145;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', 149);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', 149);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},m,[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},m,[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./m; % #of cases 
        N_test = numel(C_test{1,1}{1,1})./m; % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',20);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',20);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'AtrialFibrillation')
        m = 640;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', 645);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', 645);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,:);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,:);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'BasicMotions')
        m = 100;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', 105);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', 105);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',10);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',10);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'CharacterTrajectories')
        X_train = [];
        X_test = [];
        Y_train = [];
        Y_test = [];
        N_train = [];
        N_test = [];
        display("unequal length TS not supported")
    elseif strcmp(folder_name,'Cricket')

        m = 1197;
        for j = 1:6
            testdata = arffFiles(2*(j)).name;
            traindata = arffFiles(2*(j)+1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},m+1,[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},m+1,[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,50:1:1000),'gaussian',4);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,50:1:1000),'gaussian',4);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'DuckDuckGeese')
        m = 270;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            testdata = arffFiles(2*(j)).name;
            traindata = arffFiles(2*(j)+1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end),'gaussian',10);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end),'gaussian',10);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'ERing')
        m = 65;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},m+1,[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},m+1,[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,1:end-1);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,1:end-1);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'EigenWorms')
        m = 17984;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},m+1,[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},m+1,[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',200);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',200);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'Epilepsy')
        m = 206;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,:);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,:);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'EthanolConcentration')
        m = 1751;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            testdata = arffFiles(2*(j)).name;
            traindata = arffFiles(2*(j)+1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,1:1:end);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,1:1:end);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'FaceDetection')
        % Done but skip this for now... it took a while to run
        m = 62;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            % read for labels
            count = 1;
            fid = fopen(traindata);
                C_train_running = textscan(fid, '%q','HeaderLines', 78);
            fclose(fid);

            Xtrain = {};
            Xtest = {};
            for k = 1:size(C_train_running{1,1},1)
                C_train_running{1,1}{k,1} = ...
                    cellfun(@str2num,regexprep(strsplit(strrep(C_train_running{1,1}{k,1},'\n',','),{','}),"'",'').');
                C_train_tmp = reshape(C_train_running{1,1}{k,1}(1:end-1),62,[]);
                C_train_label_tmp = C_train_running{1,1}{k,1}(end);

                Xtrain{k} = C_train_tmp;
                Y_train = [Y_train;C_train_label_tmp];
            end

            fid = fopen(testdata);
                C_test_running = textscan(fid, '%q','HeaderLines', 78);
            fclose(fid);

            for k = 1:size(C_test_running{1,1},1)
                C_test_running{1,1}{k,1} = ...
                    cellfun(@str2num,regexprep(strsplit(strrep(C_test_running{1,1}{k,1},'\n',','),{','}),"'",'').');
                C_test_tmp = reshape(C_test_running{1,1}{k,1}(1:end-1),62,[]);
                C_test_label_tmp = C_test_running{1,1}{k,1}(end);

                Xtest{k} = C_test_tmp;
                Y_test = [Y_test;C_train_label_tmp];
            end
        end

        
        N_train = size(C_train_running{1,1},1); % #of cases 
        N_test = size(C_test_running{1,1},1); % #of cases
        
        clear X_train
        for k = 1:size(Xtrain,2) % case #
            X_train(:,:,k) = Xtrain{k}.';
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for k = 1:size(Xtest,2) % case #
            X_test(:,:,k) = Xtest{k}.';
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'FingerMovements')
        m = 50;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',10);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',10);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'HandMovementDirection')
        m = 400;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',15);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',15);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'Handwriting')
        m = 152;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smooth(Xtrain{j}(k,1:end-1));
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smooth(Xtest{j}(k,1:end-1));
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'Heartbeat')
        m = 405;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',10);
            end
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',10);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'InsectWingbeat')
        % skip this for now... take too long
        m = 62;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            % read for labels
            count = 1;
            fid = fopen(traindata);
                C_train_running = textscan(fid, '%q','HeaderLines', 29);
            fclose(fid);

            Xtrain = {};
            for k = 1:size(C_train_running{1,1},1)
                C_tmp = strsplit(strrep(strrep(C_train_running{1,1}{k,1},'\n',','),',?',''),{','});
                C_train_running{1,1}{k,1} = ...
                    cellfun(@str2num,regexprep(C_tmp(1:end-1),"'",'').');
                C_train_label_running{k} = strtrim(C_tmp{end});
            end

            fid = fopen(testdata);
                C_test_running = textscan(fid, '%q','HeaderLines', 29);
            fclose(fid);

            Xtrain = {};
            for k = 1:size(C_test_running{1,1},1)
                C_tmp = strsplit(strrep(strrep(C_test_running{1,1}{k,1},'\n',','),',?',''),{','});
                C_test_running{1,1}{k,1} = ...
                    cellfun(@str2num,regexprep(C_tmp(1:end-1),"'",'').');
                C_test_label_running{k} = strtrim(C_tmp{end});
            end
        end

        N_train = size(C_train_running{1,1},1); % #of cases 
        N_test = size(C_test_running{1,1},1); % #of cases
        
        clear X_train
        for k = 1:size(Xtrain,2) % case #
            X_train(:,:,k) = Xtrain{k}.';
        end
        Y_train = grp2idx(Y_train);

        clear X_test
        for k = 1:size(Xtest,2) % case #
            X_test(:,:,k) = Xtest{k}.';
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'JapaneseVowels')
        X_train = [];
        X_test = [];
        Y_train = [];
        Y_test = [];
        N_train = [];
        N_test = [];
        display("unequal length TS not supported")
    elseif strcmp(folder_name,'LSST')
        m = 36;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',10);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',10);
            end
        end
        Y_test = Xtest{1,1}(:,end);

    elseif strcmp(folder_name,'Libras')
        m = 45;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smooth(Xtrain{j}(k,1:end-1));
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smooth(Xtest{j}(k,1:end-1));
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'MotorImagery')
        m = 3000;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',100);
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',100);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'NATOPS')
        m = 51;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',15);
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',15);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'PEMS-SF')
        m = 144;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,1:end-1),'gaussian',20).';
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,1:end-1),'gaussian',20).';
            end
        end
        Y_test = Xtest{1,1}(:,end);
    elseif strcmp(folder_name,'PenDigits')
        m = 8;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smooth(Xtrain{j}(k,1:end-1));
            end
        end
        Y_train = Xtrain{1,1}(:,end);
        Y_train = grp2idx(Y_train);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smooth(Xtest{j}(k,1:end-1));
            end
        end
        Y_test = Xtest{1,1}(:,end);
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'PhonemeSpectra')
        m = 217;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',5).';
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',5).';
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'RacketSports')
        m = 30;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',5).';
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',5).';
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'SelfRegulationSCP1')
        m = 896;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = smoothdata(Xtrain{j}(k,:),'gaussian',10).';
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = smoothdata(Xtest{j}(k,:),'gaussian',10).';
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'SelfRegulationSCP2')
        m = 1152;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,:);
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,:);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'SpokenArabicDigits')
        X_train = [];
        X_test = [];
        Y_train = [];
        Y_test = [];
        N_train = [];
        N_test = [];
        display("unequal length TS not supported")
    elseif strcmp(folder_name,'StandWalkJump')
        m = 2500;
        Y_train = []; Y_test = [];
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;

            thisfmt = repmat('%f ', 1, m);

            fid = fopen(traindata);
            C_train{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_train_tmp = cell2mat(C_train{1,j}(:,1:end-1));
            C_train_label_tmp = C_train{j}{1,end};
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, [thisfmt, ' %C'],'Delimiter',{','},'HeaderLines', m+5);
            C_test_tmp = cell2mat(C_test{1,j}(:,1:end-1));
            C_test_label_tmp = C_test{j}{1,end};
            fclose(fid);

            Xtrain{j} = C_train_tmp;
            Xtest{j} = C_test_tmp;
            Y_train = [C_train_label_tmp];
            Y_test = [C_test_label_tmp];
        end
        N_train = size(C_train{1,1}{1,1},1); % #of cases 
        N_test = size(C_test{1,1}{1,1},1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,:);
            end
        end
        Y_train = grp2idx(Y_train);

        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,:);
            end
        end
        Y_test = grp2idx(Y_test);
    elseif strcmp(folder_name,'UWaveGestureLibrary')
        m = 315;
        for j = 1:numfiles/2-1
            traindata = arffFiles(2*(j)).name;
            testdata = arffFiles(2*(j)-1).name;
            
            fid = fopen(traindata);
            C_train{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);
    
            fid = fopen(testdata);
            C_test{j} = textscan(fid, '%f','Delimiter',{':',','},'HeaderLines', m+5);
            fclose(fid);

            Xtrain{j} = reshape(C_train{1,j}{1,1},(m+1),[]).';
            Xtest{j} = reshape(C_test{1,j}{1,1},(m+1),[]).';
        end
        N_train = numel(C_train{1,1}{1,1})./(m+1); % #of cases 
        N_test = numel(C_test{1,1}{1,1})./(m+1); % #of cases
        
        clear X_train
        for j = 1:size(Xtrain,2) % dimension
            for k = 1:size(Xtrain{j},1) % case #
                X_train(j,:,k) = Xtrain{j}(k,1:end-1;
            end
        end
        Y_train = Xtrain{1,1}(:,end);

        clear X_test
        for j = 1:size(Xtest,2) % dimension
            for k = 1:size(Xtest{j},1) % case #
                X_test(j,:,k) = Xtest{j}(k,1:end-1);
            end
        end
        Y_test = Xtest{1,1}(:,end);
    end
end