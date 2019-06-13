
function acc_per_class = computeAcc(predict_label, true_label, classes) 
    nclass = length(classes);
    acc_per_class = zeros(nclass, 1);
    for i=1:nclass
        idx = find(true_label==classes(i));
        acc_per_class(i) = sum(true_label(idx) == predict_label(idx)) / length(idx);
    end
    acc_per_class = mean(acc_per_class);