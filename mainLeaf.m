%Primera capa %
XY=imread('xxxxx 2.png');
XY=imresize(XY,[512 512]);
imagen=rgb2gray(XY);
filtro=[1,-3,-3;-1,3,-1;0,-2,-2];%filtro invertido *****1
imagenVec=reshape(imagen,1,[]);
filtroVec=reshape(filtro,1,[]);
[imageSize,asd]=size(imagen);
[x,y]=size(filtro);
rango=(imageSize-x)+1;

%MATRIZ DE PESOS CAPA 1

vecIndexI=1:2;
vecIndexJ=1:2;
vecValue=1:2;

i=1;
index=1;
con=0;

while(i<(imageSize-2)*imageSize)
    
    con=con+1;
    j=i+imageSize;
    k=j+imageSize;

    
    vecIndexI(index)=i;
    vecIndexJ(index)=con;
    vecValue(index)=filtroVec(1);
    
    vecIndexI(index+1)=i+1;
    vecIndexJ(index+1)=con;
    vecValue(index+1)=filtroVec(2);

    vecIndexI(index+2)=i+2;
    vecIndexJ(index+2)=con;
    vecValue(index+2)=filtroVec(3);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    vecIndexI(index+3)=j;
    vecIndexJ(index+3)=con;
    vecValue(index+3)=filtroVec(4);

    vecIndexI(index+4)=j+1;
    vecIndexJ(index+4)=con;
    vecValue(index+4)=filtroVec(5);

    vecIndexI(index+5)=j+2;
    vecIndexJ(index+5)=con;
    vecValue(index+5)=filtroVec(6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    vecIndexI(index+6)=k;
    vecIndexJ(index+6)=con;
    vecValue(index+6)=filtroVec(7);

    vecIndexI(index+7)=k+1;
    vecIndexJ(index+7)=con;
    vecValue(index+7)=filtroVec(8);

    vecIndexI(index+8)=k+2;
    vecIndexJ(index+8)=con;
    vecValue(index+8)=filtroVec(9);
    index=index+9;

    
    
    if(rem(i+2,imageSize)==0)
        i=i+3;
    else
        i=i+1;
    end
end

w4=sparse(vecIndexI,vecIndexJ,vecValue); %DEFINICIÓN MATRIZ DE PESOS
a5=im2double(imagenVec);
argRelu=sparse(a5*w4); %CONVOLUCION(IMAGEN,FILTRO)
n=1; %Aumenta los valores para el input
inputCapa2=reshape(n*funcionAct(argRelu),rango,[]);
%imshow(reshape(inputCapa2,254,[]));



%MATRIZ DE PESOS CAPA 2

[imageSizeCapa2,asd]=size(inputCapa2);
filtro2=[0,3,3;-5,-5,0;-3,0,1];
filtroVec2=reshape(filtro2,1,[]);

[x,y]=size(filtro2);
rango=(imageSizeCapa2-x)+1;


vecIndexII=1:2;
vecIndexJJ=1:2;
vecValuee=1:2;

i=1;
index=1;
con=0;

while(i<(imageSizeCapa2-2)*imageSizeCapa2)
    
    con=con+1;
    j=i+imageSizeCapa2;
    k=j+imageSizeCapa2;

    
    vecIndexII(index)=i;
    vecIndexJJ(index)=con;
    vecValuee(index)=filtroVec2(1);
    
    vecIndexII(index+1)=i+1;
    vecIndexJJ(index+1)=con;
    vecValuee(index+1)=filtroVec2(2);

    vecIndexII(index+2)=i+2;
    vecIndexJJ(index+2)=con;
    vecValuee(index+2)=filtroVec2(3);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    vecIndexII(index+3)=j;
    vecIndexJJ(index+3)=con;
    vecValuee(index+3)=filtroVec2(4);

    vecIndexII(index+4)=j+1;
    vecIndexJJ(index+4)=con;
    vecValuee(index+4)=filtroVec2(5);

    vecIndexII(index+5)=j+2;
    vecIndexJJ(index+5)=con;
    vecValuee(index+5)=filtroVec2(6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    vecIndexII(index+6)=k;
    vecIndexJJ(index+6)=con;
    vecValuee(index+6)=filtroVec2(7);

    vecIndexII(index+7)=k+1;
    vecIndexJJ(index+7)=con;
    vecValuee(index+7)=filtroVec2(8);

    vecIndexII(index+8)=k+2;
    vecIndexJJ(index+8)=con;
    vecValuee(index+8)=filtroVec2(9);
    index=index+9;

    
    
    if(rem(i+2,imageSizeCapa2)==0)
        i=i+3;
    else
        i=i+1;
    end
end

w3=sparse(vecIndexII,vecIndexJJ,vecValuee); %DEFINICIÓN MATRIZ DE PESOS
inputCapa2=reshape(inputCapa2,1,[]);
[sizet,tt]=size(inputCapa2);
a4=sparse(inputCapa2);

argRelu=sparse(a4*w3); %CONVOLUCION(IMAGEN,FILTRO)
n2=300; %Aumenta los valores para el input
a3=sparse(n*funcionAct(argRelu));
%imshow(reshape(inputCapa3,rango,[]));

%POOLING

%toPool=reshape(a3,1,[]);
vecValueP=1:2;

i=1;
con=0;

while(i<(rango-1)*(rango))
    con=con+1;
    j=i+rango;
    vec=[a3(i),a3(j),a3(i+1),a3(j+1)];
    vecValueP(con)=sum(vec)/4;
    
    if rem(i+1,rango)==0
        i=i+2+rango;
    else
        i=i+2;

    end
    
end

vecIndexIP=1:con;
%imshow(reshape(2*vecValueP,sqrt(rango*rango/4),[]));
a2=sparse(2*vecValueP);
%POOLING



%FULLY CONNECTED NN LAYER
[x,y]=size(a2);
w1=sparse(randi([-3,3],y,2));
a1=sparse(funcionAct(a2*w1));
y=[1,0];
lr=0.01;
%q=sparse(reshape(inputCapa2,1,[]));
vecIndexI=0;
vecIndexII=0;
vecIndexJ=0;
vecIndexJJ=0;
vecIndexIP=0;
vecValueP=0;
vecValuee=0;
vecValue=0;

%while(dot(salida,y)>0)
    %back del pooling
    %w3=w3-lr*functionActD(a3,w3,2)*functionActD(a2,w1',1)*diag(a1-y);
    
    %z= -1*(exp(a5*w4))*w4';
    x=a5*w4;
    %%%%% Maraña
    
    k=find(x);
    [N,n]=size(x);
    Yo=zeros(1,n);
    Yo(k)=-1.*(exp(x(k))); 
    Yo=sparse(Yo*w4');
    w=(1+exp(-a5*w4))*(1+exp(-a5*w4))';        
    %%%%% Maraña
    
    
    zw=sparse((Yo*(1/w)));
    zf=sparse(functionActD(a4,w3',1));
    
    %w4p=sparse()
    %w4p=260100*262144=510*510*512*512
    w1=w1-lr*sparse(functionActD(a2',w1',2)'*diag(a1-y));
    disp('hi');
    %w4=w4-lr*(zf'*zw);  
    %dot(salida,y)
%end
















