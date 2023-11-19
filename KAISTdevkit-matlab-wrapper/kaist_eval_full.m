% kaist_eval_full
% Day and night
% day
% night
% all
% Scale
% near                 [ 115         ]
% medium         [ 45    115 ]
% far                    [           45  ]
% Occlusion
% no
% partial
% heavy

function kaist_eval_full(dtDir, gtDir, reval, writeRes)
% dtDir: detection results dir, e. g.,
% E:\pyDemo\cross-modality-det\MBNet\MBNet-master\data\result
% gtDir: improved annotation dir, e.g., 'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\KAISTdevkit-matlab-wrapper\improve_annotations_liu'

if nargin < 3, reval = true; end
if nargin < 4, writeRes = true; end

sepPos = find(dtDir=='\' | dtDir=='/'); % [3    10    29    35    48    53]
if length(dtDir) == sepPos(end)
    sepPos(end) = []; 
    dtDir(end) = [];
end
tname = dtDir(sepPos(end)+1:end);   % result

% 将所有的检测结果聚合到一起。分为 result/../result-test-all.txt, result/../result-test-day.txt和result/../result-test-night.txt
% 聚合以后的txt，每一行的格式为：originalTxtId, xmin, ymin, w, h, score
bbsNms = aggreg_dets(dtDir, reval, tname);  
% 有效的目标高度>=55像素，因此小于55像素的目标框均被视为ignore，不参与计算MR
exps = {
  'Reasonable-all',       'test-all',       [55 inf],    {{'none','partial'}}
  'Reasonable-day',    'test-day',    [55 inf],    {{'none','partial'}}
  'Reasonable-night', 'test-night', [55 inf],    {{'none','partial'}}
  'Scale=near',              'test-all',       [115 inf], {{'none'}}
  'Scale=medium',      'test-all',        [45 115],   {{'none'}}
  'Scale=far',                  'test-all',       [1 45],   {{'none'}}
%   'Occ=none',               'test-all',       [1 inf],      {{'none'}}
%   'Occ=partial',             'test-all',       [1 inf],      {{'partial'}}
%   'Occ=heavy',              'test-all',        [1 inf],     {{'heavy'}}
  };

res = [];

len_exps = size(exps);
for ie = 1:len_exps(1)
    res = run_exp(res, exps(ie,:), gtDir, bbsNms);
end

if writeRes
%     save(fullfile(dtDir, '..', ['res' tname(4:end) '.mat']), 'res');
    save(fullfile(dtDir(1:end-length(tname)), ['res' tname(4:end) '.mat']), 'res');
    fprintf('Results saved.\n');
end

end

function bbsNms = aggreg_dets(dtDir, reval, tname)
% dtDir: detection results full dir name, e.g.,  'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\data\result'
% tname: detection results base folder name, e.g., 'result'
% reval: evaluate existing detection results

% return aggregated files
% bbsNm.test-all
for cond = [{'test-all'}, {'test-day'}, {'test-night'}]
    desName = [tname '-' cond{1} '.txt'];   % 'result-test-all.txt'
%     desName = fullfile(dtDir(1:end-length('/det')), desName);
%     desName = fullfile(dtDir, '..', desName); %'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\data\result\result-test-all.txt'
    desName = fullfile(dtDir(1:end-length(tname)), desName);
    bbsNms.(sprintf('%s', strrep(cond{1}, '-', '_'))) = desName; % bbsNms.test-all = desName
    if exist(desName, 'file') && ~reval
        continue;
    end
    switch cond{1}
        case 'test-all'
            setIds = 6:11;
            skip = 20;
            vidIds = {0:4 0:2 0:2 0 0:1 0:1};
        case 'test-day'
            setIds = 6:8; 
            skip = 20;
            vidIds = {0:4 0:2 0:2};
        case 'test-night'
            setIds = 9:11;
            skip = 20;
            vidIds = {0 0:1 0:1};
    end
    fidA = fopen(desName, 'w');
    num = 0;
    for s=1:length(setIds)
        for v=1:length(vidIds{s})
            for i=skip-1:skip:99999
                detName = sprintf('set%02d_V%03d_I%05d.txt', setIds(s), vidIds{s}(v), i); % 'set06_V000_I00019.txt'
                detName = fullfile(dtDir, detName); % a detection result file, e.g., 'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\data\result\set06_V000_I00019.txt'
                if ~exist(detName, 'file')
                    continue;
                end
                num = num + 1;
                [~, x1, y1, x2, y2, score] = textread(detName, '%s %f %f %f %f %f');    % get all lines according to the format
                for j = 1:length(score)
                    fprintf(fidA, '%d,%.4f,%.4f,%.4f,%.4f,%.8f\n', num, x1(j)+1, y1(j)+1, x2(j)-x1(j), y2(j)-y1(j), score(j));
                end
            end
        end
    end
    fclose(fidA);
end

end

function res = run_exp(res, iexp, gtDir, bbsNms)
% iexp: {'Reasonable-all'}    {'test-all'}    {[55 Inf]}    {1×1 cell}
thr = .5;
mul = 0;
ref = 10.^(-2:.25:0); % 0.0100    0.0178    0.0316    0.0562    0.1000    0.1778    0.3162    0.5623    1.0000
% ilbls 表示ignore labels. 因此只考虑person类别
pLoad0={'lbls',{'person'},'ilbls',{'people','person?','cyclist'}}; % {'lbls'}    {1×1 cell}    {'ilbls'}    {1×3 cell}
% .hRng     - [] range of acceptable obj heights
% .xRng     - [] range of x coordinates of bb extent
% .yRng     - [] range of y coordinates of bb extent
% .vRng     - [] range of acceptable obj occlusion levels
% 上述定义请参考bbGt.bbLoad
pLoad = [pLoad0, 'hRng',iexp{3}, 'vType',iexp{4},'xRng',[5 635],'yRng',[5 507]];

res(end+1).name = iexp{1};

bbsNm = bbsNms.(sprintf('%s',strrep(iexp{2},'-','_'))); % 取文件名：'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\data\result\..\result-test-all.txt'
%% 下面的代码在于实现：
% 1. loadAll: 加载GT(ground-truth)和DT(detection)Boxes
% 2. evalRes: 按照modified criteria(含ignore)评价gt/dt boxes是否匹配，并返回tp, fp, fn
% 3. compRoc: 计算ROC或者PR. roc flag来判断该计算ROC还是PR. 
%    ROC返回(xs: fppi, ys: tp); PR返回(xs:Precision, ys: Recall)
% 4. mr =  1 - recall

% original annotations
annoDir = fullfile(gtDir,iexp{2},'annotations');    %'E:\pyDemo\cross-modality-det\MBNet\MBNet-master\KAISTdevkit-matlab-wrapper\improve_annotations_liu\test-all\annotations'
[gt,dt] = bbGt('loadAll',annoDir,bbsNm,pLoad); % 加载所有的gtBoxes和detBoxes,并将结果分别返回成cell格式，每个cell都是一个图片中的所有框。一个cell中每一行的格式为[xmin, ymin, w, h, score]
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);% 评价dt中有多少FP, 评价gt中有多少FN. mul表示一个dt与多个gt匹配
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,ref); % Compute ROC(roc=1: TP vs FPPI) or PR(roc=0: P vs R) based on outputs of evalRes on multiple images.更多请看compROC
miss_ori=exp(mean(log(max(1e-10,1-miss))));
roc_ori=[score fp tp];

res(end).ori_miss = miss;
res(end).ori_mr = miss_ori;
res(end).roc = roc_ori;

% improved annotations
annoDir = fullfile(gtDir,iexp{2}, 'annotations_KAIST_test_set');
[gt,dt] = bbGt('loadAll',annoDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,ref);
miss_imp=exp(mean(log(max(1e-10,1-miss))));
roc_imp=[score fp tp];

res(end).imp_miss = miss;
res(end).imp_mr = miss_imp;
res(end).imp_roc = roc_imp;

fprintf('%-30s \t log-average miss rate = %02.2f%% (%02.2f%%) recall = %02.2f%% (%02.2f%%)\n', iexp{1}, miss_ori*100, miss_imp*100, roc_ori(end, 3)*100, roc_imp(end, 3)*100);

end
