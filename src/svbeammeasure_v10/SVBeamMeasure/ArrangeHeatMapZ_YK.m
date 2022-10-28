cd '\\MA2FILES\Production systems\SN1022\argolight profiles\2020-06-12 Argolight\HeatMap'

fname_date = 'SN1022, 2020-06-12, Ch';
% lambda = [488, 532, 595, 647];
lambda = [595, 647];
% exposure = [200, 80, 500, 300];
exposure = [500, 300];
% power = [500, 2000, 2000, 2000];
power = [2000, 2000];

for ind_lambda = 1:2
    
    img_map = [];
    
    for ind_map = 1:1
        
        img_ref = [];
        
        for ind_ch = 1:4
%             fname = [fname_date num2str(ind_ch) ', Ref, ' num2str(lambda(ind_lambda))...
%                 ' ' num2str(power(ind_lambda)) 'mw@' num2str(exposure(ind_lambda)) 'ms _HeatMap' num2str(ind_map) '.jpg'];
            fname = [fname_date num2str(ind_ch) ', Ref, ' num2str(lambda(ind_lambda))...
                ' ' num2str(power(ind_lambda)) 'mw@' num2str(exposure(ind_lambda)) 'ms , LM_HeatMap1Gauss.png'];
            img = imread(fname);
            img_ref = [img_ref, img];
        end

        img_test = [];

        for ind_ch = 1:4
%             fname = [fname_date num2str(ind_ch) ', Test, ' num2str(lambda(ind_lambda))...
%                 ' ' num2str(power(ind_lambda)) 'mw@' num2str(exposure(ind_lambda)) 'ms _HeatMap' num2str(ind_map) '.jpg'];
            fname = [fname_date num2str(ind_ch) ', Test, ' num2str(lambda(ind_lambda))...
                ' ' num2str(power(ind_lambda)) 'mw@' num2str(exposure(ind_lambda)) 'ms , LM_HeatMap1Gauss.png'];
            img = imread(fname);
            img_test = [img_test, img];
        end

        img_map = [img_map; img_ref; img_test];

    end
    
    figure; imagesc(img_map), axis image
    fname_save = [fname_date ' ' num2str(lambda(ind_lambda))...
                ' ' num2str(power(ind_lambda)) 'mw@' num2str(exposure(ind_lambda)) 'ms, LM_HeatMap1Gauss.png'];
    imwrite(img_map,fname_save)
    
end


% cd '\\ma\Production systems\SN1019 (hi-power)\2020-05-04 Argolight'
% 
% fname_date = '05-04-20 Ch';
% lambda = [473, 532, 595, 647];
% % exposure = [80, 500, 300];
% 
% 
% for ind_lambda = 4:4
%     
%     img_map = [];
%     
%     for ind_map = 1:2
%         
%         img_ref = [];
%         
%         for ind_ch = 1:4
%             fname = [fname_date num2str(ind_ch) ' Ref ' num2str(lambda(ind_lambda))...
%                 ' _HeatMap' num2str(ind_map) '.png'];
%             img = imread(fname);
%             img_ref = [img_ref, img];
%         end
% 
%         img_test = [];
% 
%         for ind_ch = 1:4
%             fname = [fname_date num2str(ind_ch) ' Test ' num2str(lambda(ind_lambda))...
%                 ' _HeatMap' num2str(ind_map) '.png'];
%             img = imread(fname);
%             img_test = [img_test, img];
%         end
% 
%         img_map = [img_map; img_ref; img_test];
% 
%     end
%     
%     figure; imagesc(img_map), axis image
%     fname_save = [fname_date ' ' num2str(lambda(ind_lambda))...
%                 ' _HeatMap.jpg'];
%     imwrite(img_map,fname_save)
%     
% end
