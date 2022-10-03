from kivymd.uix.button import MDFlatButton ,MDIconButton,MDFloatingActionButton
from kivy.properties import ObjectProperty,StringProperty
from kivy.uix.screenmanager import Screen,ScreenManager
from kivy.uix.screenmanager import Screen,ScreenManager
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import StringProperty
from kivy.uix.boxlayout import  BoxLayout
from kivy.core.window import Window
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from os.path import dirname, join
from kivy.utils import platform
from datetime import datetime
from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.app import App
import os
import sqlite3
from keras.preprocessing.image import img_to_array,load_img,smart_resize
import numpy as np
from labels import labels
from datetime import datetime
import tensorflow as tf

class GetStarted(Screen):
    pass

class Intro(Screen):
    pass

class Gallery(Screen):
    def on_pre_enter(self):
        if platform=='android':       
            self.ids.filechooser.path=join(os.getenv('EXTERNAL_STORAGE'),"DCIM")
        else:
            self.ids.filechooser.path="test"
    def selected(self,filename):
        return filename[0]
    


class CameraClick(Screen):
    def take_selfie(self, *args):
        app=MDApp.get_running_app()
        self.camera = self.ids['camera']
        if platform=='android':        
            img_name="image_"+str(app.date_time)+".png"
            img_path=join(os.getenv('EXTERNAL_STORAGE'),"Guarden_images")
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            location =join(img_path,img_name)
            self.camera.export_to_png(location)
            return location
        else:
            img_name="image_"+str(app.date_time)+".png"
            img_path="Guarden_images"
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            location=join(img_path,img_name)
            self.camera.export_to_png(location)
            return location
        

class Output(Screen):
    pass

class Result(Screen):
    is_leaf=0    
    img_name=""
    def on_pre_enter(self):
        self.predict_from_tflite()
    def preprocess(self,img_size):
        app=MDApp.get_running_app()
        img=load_img(self.manager.img_name)
        img=img_to_array(img)
        img=smart_resize(img,(img_size,img_size))
        img=np.array(img)
        img=img/255
        img=img.reshape(1,img_size,img_size,3)
        return img
             
    def identify(self):
        img=self.preprocess(128)
        img=np.array(img,dtype="float32")
        main_model=tf.lite.Interpreter('models/json_plant_identifier_model.tflite')
        main_model.allocate_tensors()
        inputs=main_model.get_input_details()
        outputs=main_model.get_output_details()
        input_shape=inputs[0]['shape']
        main_model.set_tensor(inputs[0]['index'],img)
        main_model.invoke()
        output_data=main_model.get_tensor(outputs[0]['index'])
        pred=list(output_data)
        
        self.is_leaf=np.argmax(pred,axis=-1)
        #self.is_leaf=np.argmax(lnl_model.predict(img),axis=-1)
        return self.is_leaf
    
    def get_image(self,img_n):
        self.ids.img.source= f"disease_images/{img_n}.jpg"

    def predict_from_tflite(self):
        if self.identify()==1:
            img=self.preprocess(64)
            img=np.array(img,dtype="float32")
            main_model=tf.lite.Interpreter("models/kag_guarden.tflite")
            main_model.allocate_tensors()
            inputs=main_model.get_input_details()
            outputs=main_model.get_output_details()
            input_shape=inputs[0]['shape']
            main_model.set_tensor(inputs[0]['index'],img)
            main_model.invoke()
            output_data=main_model.get_tensor(outputs[0]['index'])
            pred=list(output_data)
            out=np.argmax(pred,axis=-1)
            f=labels[out[0]]
            con=sqlite3.connect("details.db")
            c=con.execute(f"SELECT * from diseases where name='{f}'")
            l=c.fetchall()
            self.get_image(f)
            print("returned label ----->",f)
            print(l)
            con.close()
            self.ids.title.text=l[0][0]
            self.ids.typ.text=l[0][1]
            self.ids.sym.text=l[0][2]
            self.ids.treat.text=l[0][3]
            
        else:
            self.manager.current="not_leaf"
            
        
    
    
class Not_Leaf(Screen):
    pass
    
class ScreenManagement(ScreenManager):
    MY_GLOBAL = StringProperty('')
    img_name = StringProperty('')


    

class CameraApp(MDApp):
    date_time =datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    sm=ScreenManager()
    sm.add_widget(GetStarted(name='get'))
    sm.add_widget(Intro(name='intro'))
    sm.add_widget(CameraClick(name='cam'))
    sm.add_widget(Gallery(name='gal'));
    sm.add_widget(Output(name='output'))
    sm.add_widget(Not_Leaf(name='not_leaf'))
    sm.add_widget(Result(name='res'))
    def build(self):
        self.icon = 'logos/icon_new.png'
        self.title= "GUARDEN"
        if platform=='android':
            #from cameraxf import CameraXF
            from android.permissions import request_permissions, Permission,check_permission
            request_permissions([
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
            ])
        b=Builder.load_file("helper.kv")
        return b
    
if __name__=='__main__':
    CameraApp().run()
        
