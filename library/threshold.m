function[F]=threshold(X,tau)
[row,col]=size(X);
F=zeros(row,col);
for i=1:row
    for j=1:col
        if(X(i,j)>0)
            F(i,j)=max(X(i,j)-tau,0);
        else if(X(i,j)==0)
                F(i,j)=0;
            else
                F(i,j)=-1*max(-X(i,j)-tau,0);
            end
        end
        
    end
end

end