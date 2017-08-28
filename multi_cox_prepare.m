% This function is used to prapare the training and testing file for
% multi_task survival analysis. Beside cross validation it will also 
% do preprocess for Cox model

function multi_cox_prepare(folder, num_cv)
%   Detailed explanation goes here
    %clear all;
    n_num_cv=str2double(num_cv);
    current_path=cd;
    cd(strcat(current_path,folder)); %'/Noname_addone_miRNA_use/'
    dd=dir('*.csv');
    fileNames = {dd.name}; 
    ntask=numel(fileNames);
    data = cell(ntask,3);
    data(:,1) = regexprep(fileNames, '.csv','');

    mkdir(strcat(cd,'/cv'))
    for ii = 1:ntask
       %disp(ii);
       data{ii,2} = dlmread(fileNames{ii});
       cvf = crossvalind('Kfold',data{ii,2}(:,2),n_num_cv); % devide into train and test
       data{ii,3} = cvf;
       csvwrite(strcat(cd,'/cv/cv_',num2str(ii),'.csv'),cvf)
    end
    
    %% merge and cv
    
    for k =1:n_num_cv
        train_cell=cell(ntask,1);
        test_cell=cell(ntask,1);
        for j=1:ntask
            test = (data{j,3} == k); 
            train = (data{j,3} ~= k);
            test_cell{j}=data{j,2}(test,:);
            train_cell{j}=cox_preprocess(data{j,2}(train,3:end),data{j,2}(train,1),'censoring',~(data{j,2}(train,2)));
        end
        save(strcat(cd,'/train_',num2str(k),'.mat'),'train_cell');
        save(strcat(cd,'/test_',num2str(k),'.mat'),'test_cell');
    end
    cd(current_path);
    
    
    
    
    
    
    
    %% cox_preprocess
        function [Processed] = cox_preprocess(X,y,varargin)
    %   [...] = cox_preprocess(X,Y,'PARAM1',VALUE1,'PARAM2',VALUE2,...) specifies
    %   additional parameter name/value pairs chosen from the following:
    %
    %      Name          Value
    %      'baseline'    The X values at which the baseline hazard is to be
    %                    computed.  Default is mean(X), so the hazard at X is
    %                    h(t)*exp((X-mean(X))*B).  Enter 0 to compute the
    %                    baseline relative to 0, so the hazard at X is
    %                    h(t)*exp(X*B).
    %      'censoring'   A boolean array of the same size as Y that is 1 for
    %                    observations that are right-censored and 0 for
    %                    observations that are observed exactly.  Default is
    %                    all observations observed exactly.
    %      'frequency'   An array of the same size as Y containing non-negative
    %                    integer counts.  The jth element of this vector
    %                    gives the number of times the jth element of Y and
    %                    the jth row of X were observed.  Default is 1
    %                    observation per row of X and Y.
    %      'init'        A vector containing initial values for the estimated
    %                    coefficients B.
    %      'options'     A structure specifying control parameters for the
    %                    iterative algorithm used to estimate B.  This argument
    %                    can be created by a call to STATSET.  For parameter
    %                    names and default values, type STATSET('coxphfit').


    narginchk(2,inf);
    % Check the required data arguments
    if ndims(X)>2 || ~isreal(X)
        error(message('stats:coxphfit:BadX'));
    end
    if ~isvector(y) || ~isreal(y)
        error(message('stats:coxphfit:BadY'));
    end

    % Process the optional arguments
    okargs =   {'baseline' 'censoring' 'frequency' 'init' 'options'};
    defaults = {[]         []          []          []     []};
    [baseX cens freq init options] = internal.stats.parseArgs(okargs,defaults,varargin{:});

    if ~isempty(cens) && (~isvector(cens) || ~all(ismember(cens,0:1)))
        error(message('stats:coxphfit:BadCensoring'));
    end
    if ~isempty(freq) && (~isvector(freq) || ~isreal(freq) || any(freq<0))
        error(message('stats:coxphfit:BadFrequency'));
    end
    if ~isempty(baseX) && ~(isnumeric(baseX) && (isscalar(baseX) || ...
                                 (isvector(baseX) && length(baseX)==size(X,2))))
        error(message('stats:coxphfit:BadBaseline'));
    elseif isscalar(baseX)
        baseX = repmat(baseX,1,size(X,2));
    end


    % Sort by increasing time
    [sorty,idx] = sort(y);
    X = X(idx,:);
    [n,p] = size(X);
    if isempty(cens)
        cens = false(n,1);
    else
        cens = cens(idx);
    end
    if isempty(freq)
        freq = ones(n,1);
    else
        freq = freq(idx);
    end

    % Determine the observations at risk at each time
    [~,atrisk] = ismember(sorty,flipud(sorty), 'legacy');
    atrisk = length(sorty) + 1 - atrisk;     % "atrisk" used in nested function
    tied = diff(sorty) == 0;
    tied = [false;tied] | [tied;false];      % "tied" used in nested function

    Processed = struct('X',X,'freq',freq,'cens',cens,'atrisk',atrisk);

    end



end



