import math
import sys
import random

class Swing:

    cN=["2XS","XS", "Sv", "Sr", "M", "L"] 
    cNM = [1.21, 1.41, 1.61, 1.61, 1.81, 2.01] 
    cNG = [7.47, 7.77, 8.12, 8.12, 8.56, 9.32] 

    h=0.001   #ミリ秒で系を更新する

    G = 9.80665
    RAD= math.pi/180  
    DEG=180/ math.pi   

    M = 50   
    Lh = 1.58  
    m1=0.626*M  
    m2=0.246*M  
    m3=0.128*M   
    L1=0.501*Lh  
    L2=0.249*Lh  
    L3=0.25*Lh   
    a=L2*(m3+0.5*m2)/M  
    b=L2-a  


    def __init__(self,eps = -0.006, chn = 4,x=0, phi_init=-20, delta_phi_0=15,
                     VR=0, coef_Hooke=1):
                  #  外力の反映 1:2XS 2:XS 3:Sv 4:Sr 5:M 6:L,初期座面角(deg)
                  #                  ,初期上体角(振動中心からの),上体振動の中心(deg)
                  # 何往復 , 座面の前進局面での限界値 , VR=0:Realブランコ

        self.eps = eps 
        self.chn = chn -1 # matlabの配列は1から,pythonは0から。その補正!
        self.x_init = -x*self.RAD
        r=random.uniform(0.9, 1.1) #少しのランダム性
        self.x_init = -r*x*self.RAD

        self.delta_phi_0 = delta_phi_0 *self.RAD
        self.phi_init = phi_init *self.RAD 
        #self.nmax = nmax  
        #self.xDegmax = xDegmax 
        self.VR = VR
        self.coef_Hooke = coef_Hooke
        self.torq_ai = 0

        self.L =self.cNM[self.chn]  
        self.m0=self.cNG[self.chn]  


        self.coef_iner=self.eps*int(not bool(VR)) #慣性力トルクの係数 VR=0でnonzero
        self.coef_cent=self.eps*int(not bool(VR)) #遠心力トルク VR=0でnonzero
        self.coef_grav=self.eps     #重力トルク

        self.Res_sw2 =0.  #0.4*0.3   #座面の速度2乗に比例する抵抗
        self.Res_sw1 =0.  #座面の速度に比例する抵抗
        self.Res_sw0 =0.  #0.4*0.7   #速度に依存しない抵抗

        self.omega_phi= math.sqrt(self.G/(self.L-0.5*self.L1))  




    def reset(self):
        r=random.uniform(0.9, 1.1) #少しのランダム性
        self.phi=r*self.phi_init* math.cos(0.1* math.pi) + self.delta_phi_0 #体初期角
        self.d_phi=-r*self.omega_phi*self.phi_init* math.sin( 0.1* math.pi) #初期角速度
        self.z=1.e-60 
        self.z00=1. 
        self.t= self.t1= self.t2 =0 
        self.n = 0
        self.x = self.x_init

        self.oldz= self.dz= self.oldd_phi= self.d2_phi=0 
        self.psi= self.d_psi= self.d2_psi=0 

        self.phimax= self.xmax= -99 
        self.phimin= self.xmin =99 

        self.MLB= self.atMLB= self.arg= self.torq_LB= self.torq_Hooke=0 
        self.E=self.Esw=self.torq_iner= self.torq_grav=torq_cent=0  

        #self.Tp=2* math.pi/self.omega_phi 

        self.data_csv = "t,x,z,dz,phi,d_phi,d2_phi,T,torq_LB,torq_Hooke,"\
                        "torq_iner,torq_grav,torq_cent,E,Esw\n"
        print((f"iner:{self.coef_iner},cent:{self.coef_cent},grav:{self.coef_grav}"
               f",Hooke:{self.coef_Hooke},omega:{self.omega_phi:.2f}"))

    def f(self,q,dq):
        return dq


    def  gLB(self,phi,d_phi,u):   #上体の運動方程式 , u が AI から渡されるトルク
        m0,M,m1,m2,m3,G =self.m0,self.M,self.m1,self.m2,self.m3,self.G
        L,L1,L2,L3,a,b =self.L,self.L1,self.L2,self.L3,self.a,self.b
        omega_phi,delta_phi_0 = self.omega_phi,self.delta_phi_0
        coef_iner,coef_cent,coef_grav=self.coef_iner,self.coef_cent,self.coef_grav
        x,z,dz=self.x,self.z,self.dz
        coef_Hooke=self.coef_Hooke 
        
        c_phi= math.cos(phi)
        s_phi= math.sin(phi)
        c_x  = math.cos(x)
        s_x  = math.sin(x)
        iner = (-2*L1/3 +L*c_phi -a*s_phi)*dz
        cent = ( L*s_phi+a*c_phi)*z**2
        grav =  math.sin( x*int(self.VR!=1) +phi )*G  #VRが1でない Gsin(x+\phi),1のときGsin\phi
        Hooke = -1.*omega_phi**2*( phi-delta_phi_0 )*L1/1.5  # 変位に比例する復元力
        # トルク合算
        acc= (coef_iner*iner+coef_cent*cent+coef_grav*grav+coef_Hooke*Hooke+u)*1.5/L1 

        self.torq_LB = acc *0.5*m1*L1 *L1 /1.5 
        self.torq_Hooke = coef_Hooke*Hooke*0.5*m1*L1
        self.torq_iner = coef_iner*iner*0.5*m1*L1 
        self.torq_cent = coef_cent*cent*0.5*m1*L1
        self.torq_grav = coef_grav*grav*0.5*m1*L1 
        self.torq_ai   = u*0.5*m1*L1
        return acc; 



    def g(self,x,z):
        m0,M,m1,m2,m3,G =self.m0,self.M,self.m1,self.m2,self.m3,self.G
        L,L1,L2,L3,a,b =self.L,self.L1,self.L2,self.L3,self.a,self.b
        omega_phi,delta_phi_0 = self.omega_phi,self.delta_phi_0
        Res_sw2,Res_sw1,Res_sw0=self.Res_sw2,self.Res_sw1,self.Res_sw0
        phi,d_phi,d2_phi,psi,d_psi,d2_psi=self.phi,self.d_phi,self.d2_phi,self.psi,self.d_psi,self.d2_psi

        c_psi= math.cos(psi)
        s_psi= math.sin(psi)
        c_phi= math.cos(phi)
        s_phi=math.sin(phi)
        c_x  =math.cos(x)
        s_x  =math.sin(x)
        I0 = (m0/3.+M)*L**2 +m1*L1**2/3.+m2*b**3/(3*L2) +m2*a**3/(3*L2)+m3*b**2 +m3*L3**2/3. +m1*a**2
        Ip = m1*L1*(-L*c_phi+a*s_phi) + m3*L3*(L*c_psi+b*s_psi)
        Np = m1*L1*d_phi*( L*s_phi+a*c_phi)*z+0.5*m1*L1*d_phi**2*( L*s_phi+a*c_phi) \
            +m3*L3*d_psi*(-L*s_psi+b*c_psi)*z +0.5*m3*L3*d_psi**2*(-L*s_psi+b*c_psi)
        Nd = m1*L1**2*d2_phi/3. +0.5*m1*L1*(-L*c_phi+a*s_phi)*d2_phi \
            +m3*L3**2*d2_psi/3.+0.5*m3*L3*(L*c_psi+b*s_psi)*d2_psi
        N0 = (0.5*m0+M)*L*s_x*G  + (-m1*a+m3*b+0.5*m2*(b**2-a**2)/L2)*c_x*G
        Ng = -0.5*m1*L1*math.sin(x+phi)*G  + 0.5*m3*L3*math.sin(x+psi)*G
        Er = Res_sw2*math.copysign(1,z)*z*z*L*L*L +Res_sw1*z*L*L + math.copysign(1,z)*Res_sw0
        acc = -(Np+Nd+N0+Ng+Er)/(I0+Ip)
        return acc



    def LB_rk4(self,u):
        k1 = self.h*  self.f(self.phi ,self.d_phi ) 
        c1 = self.h*self.gLB(self.phi ,self.d_phi,u) 
        k2 = self.h*  self.f(self.phi+0.5*k1 ,self.d_phi+0.5*c1)
        c2 = self.h*self.gLB(self.phi+0.5*k1 ,self.d_phi+0.5*c1,u)
        k3 = self.h*  self.f(self.phi+0.5*k2 ,self.d_phi+0.5*c2)
        c3 = self.h*self.gLB(self.phi+0.5*k2 ,self.d_phi+0.5*c2,u)
        k4 = self.h*  self.f(self.phi+k3     ,self.d_phi+c3)
        c4 = self.h*self.gLB(self.phi+k3     ,self.d_phi+c3,u)
        self.phi += (k1 + 2.*k2 + 2.*k3 + k4)/6.
        self.d_phi += (c1 + 2.*c2 + 2.*c3 + c4)/6.
        self.d2_phi=(c1 + 2.*c2 + 2.*c3 + c4)/(self.h*6.)
        #print(f"a {self.phi,self.d_phi,self.d2_phi,self.x,self.z,self.dz}")



    def SW_rk4(self):
        k1 = self.h*self.f(self.x ,self.z)  
        c1 = self.h*self.g(self.x ,self.z)    
        k2 = self.h*self.f(self.x+0.5*k1 ,self.z+0.5*c1)  
        c2 = self.h*self.g(self.x+0.5*k1 ,self.z+0.5*c1)   
        k3 = self.h*self.f(self.x+0.5*k2 ,self.z+0.5*c2)  
        c3 = self.h*self.g(self.x+0.5*k2 ,self.z+0.5*c2)  
        k4 = self.h*self.f(self.x+k3     ,self.z+c3)   
        c4 = self.h*self.g(self.x+k3     ,self.z+c3)
        self.x += (k1 + 2.*k2 + 2.*k3 + k4)/6. 
        self.z += (c1 + 2.*c2 + 2.*c3 + c4)/6.   
        self.dz=(c1 + 2.*c2 + 2.*c3 + c4)/(self.h*6.)
        #print(f"b {self.phi,self.d_phi,self.d2_phi,self.x,self.z,self.dz}")


    def observe(self):
        m0,M,m1,m2,m3,G =self.m0,self.M,self.m1,self.m2,self.m3,self.G
        L,L1,L2,L3,a,b =self.L,self.L1,self.L2,self.L3,self.a,self.b
        phi,d_phi,d2_phi,psi,d_psi,d2_psi=self.phi,self.d_phi,self.d2_phi,self.psi,self.d_psi,self.d2_psi
        x,z,dz=self.x,self.z,self.dz
        c_psi= math.cos(psi)
        s_psi= math.sin(psi)
        c_phi= math.cos(phi)
        s_phi=math.sin(phi)
        c_x  =math.cos(x)
        s_x  =math.sin(x)

        Esw = m0*L**2*z**2/6.-0.5*m0*L*c_x*G #ブランコだけ

        E = 0.5*((m0/3.+M)*L**2 +m1*(L1**2/3.+a**2) +m2*(a**2-a*b+b**2)/3. +m3*(L3**2/3.+b**2))*z**2\
           +0.5*m1*L1*(  L1*(2*z*d_phi+d_phi**2)/3.+(-L*c_phi+a*s_phi)*(z**2+z*d_phi) )\
           +0.5*m3*L3*(  L3*(2*z*d_psi+d_psi**2)/3.+( L*c_phi+b*s_phi)*(z**2+z*d_psi) )\
           -(0.5*m0+M)*L*c_x*G +0.5*m1*L1*math.cos(x+phi)*G -0.5*m3*L3*math.sin(x+psi)*G
        return Esw

