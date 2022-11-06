function a = getRandomElements(a, k, withReplacement, keepOrder)
%GETRANDOMELEMENTS Random elements of a vector, with or without replacement.
%   B = GETRANDOMELEMENTS(A, K) returns a 1 x K vector B containing N
%   random elements of the array A.
%
%   By default, only allow replacement when K > length(A).

%   Author: Damien Teney

assert(k >= 0);
if isempty(a), return; end
if isinf(k), return; end
n = numel(a);

if nargin < 3 || isempty(withReplacement), withReplacement = (k > n); end % By default, only allow replacement when k > length(a)
if nargin < 4, keepOrder = false; end % By default, allow shuffling the elements

if (k > n)
  assert(withReplacement); % If we need more sample than input elements, we MUST allow replacement
  ids = randi(n, 1, k); % Sample with replacement
else
  ids = randperm(n);
  ids = ids(1:k); % Sample WITHOUT replacement (unique IDs)
end
if keepOrder
  ids = sort(ids);
end
a = a(ids);
