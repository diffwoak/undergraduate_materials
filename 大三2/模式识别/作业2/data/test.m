load('faces.mat');
[coeff, score, ~, ~, explained] = pca(X);

figure;
scatter(score(:,1), score(:,2));
title('PCA of myVar');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

figure;
pareto(explained);
title('Explained Variance by Principal Components');
xlabel('Principal Components');
ylabel('Variance Explained (%)');
