#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 08:31:17 2021

@author: sebanb

"""

#----------------------------------------------------------------------------
#Libraries
#---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
#Class Definition
#---------------------------------------------------------------------------

#Superclass
class DynamicSystem:
    #Initiation
    def __init__(self, f, c, name):
        self.f = f # Function is defined
        self.c = c # Parameters' vector 
        self.name = name #Function's name, must be string
    # Integration in time
    def TimeEvolution(self, X0, t0, tf, N):
        T = np.linspace(t0,tf,N) # Time's values
        #T = np.arange(self.t0, self.tf+self.dt, seld.dt)
        sol = odeint(self.f,X0,T,args=(self.c,)) 
        return [T,sol]
 #User  can choose the Lorenz attractor
class Lorenz(DynamicSystem):    
    @classmethod
    def fun(cls,X,t,c):
        #sigma=c[0]
        #rho=c[1]
        #beta=c[2]
        x_dot = c[0]*(X[1] - X[0])
        y_dot = c[1]*X[0] - X[1] - X[0]*X[2]
        z_dot = X[0]*X[1] -c[2]*X[2]
        return [x_dot, y_dot, z_dot]
    def __init__(self, sigma, rho, beta, name = 'Lorenz'):
        self.sigma = sigma # Devuelve parámetro sigma
        self.rho = rho # Devuelve parámetro rho
        self.beta = beta # Devuelve parámetro beta
        self.c = [sigma, rho, beta] # Devuelve lista de parámetros params
        self.f =self.fun
        super().__init__(self.f,self.c, name = 'Lorenz')
class Duffing(DynamicSystem):
    @classmethod
    def fun(cls,X,t,c):
         dx = X[1]
         dy = -c[2] * X[1] - c[0] * X[0] - c[1] * X[0]**3 + c[3] * np.cos(c[4] * t)
         return [dx, dy]
    def __init__(self, alpha, beta, delta, gamma, omega):
        self.alpha = alpha # Devuelve parámetro alpha
        self.beta = beta # Devuelve parámetro beta
        self.delta = delta # Devuelve parámetro delta
        self.gamma = gamma # Devuelve parámetro gamma
        self.omega = omega # Devuelve parámetro omega
        self.c = [alpha, beta, delta, gamma, omega] # Devuelve lista de parámetros params
        self.f = self.fun
        super().__init__(self.f,self.c, name = 'Duffing')
class PlotSystem:
      def __init__(self,t,X):
          self.t=t # Time vector
          self.X=X #Variable data
      def plotxt(self,xname = "$x$", yname = "$t$"):
          fig1,ax=plt.subplots()
          if self.X.ndim ==1:             
                ax.plot(self.t,self.X)
          else:
                ax.plot(self.t,self.X[:,0]) 
          ax.set_xlabel(xname,size=12)
          ax.set_ylabel( yname,size=12)
          ax.grid()
          return fig1
      def plotyt(self,xname = "$y$"):
          fig1,ax=plt.subplots()
          if self.X.ndim ==1:             
                raise ValueError("For unidimensional case use .plotxt")
          else:
                ax.plot(self.t,self.X[:,1]) 
          ax.set_xlabel("$t$",size=12)
          ax.set_ylabel( xname,size=12)
          ax.grid()
          return fig1
      def plotxy(self,xname= "$x$", yname = "y"):
          fig1,ax=plt.subplots()
          ax.plot(self.X[:,0],self.X[:,1]) 
          ax.set_xlabel(xname,size=12)
          ax.set_ylabel( yname,size=12)
          ax.grid()
          return fig1
      def plotzt(self,xname = "$z$"):
          fig1,ax=plt.subplots()
          if self.X.ndim ==1:             
                raise ValueError("For unidimensional case use .plotxt")
          elif (self.X.ndim ==2 and self.X.shape[1]==2):
                raise ValueError("For bidimensional case use .plotxt and .plotyt")
          else:
                ax.plot(self.t,self.X[:,2]) 
          ax.set_xlabel("$t$",size=12)
          ax.set_ylabel( xname,size=12)
          ax.grid()
          return fig1
      def plotxyz(self, xname="$x$", yname="$y$", zname="z"):
          fig = plt.figure()
          ax = plt.axes(projection ='3d')
          ax.plot3D(self.X[:,0],self.X[:,1] , self.X[:,2])   
          ax.set_xlabel(xname, size=13)
          ax.set_ylabel(yname, size=13)
          ax.set_zlabel(zname, size=13)
          return fig
