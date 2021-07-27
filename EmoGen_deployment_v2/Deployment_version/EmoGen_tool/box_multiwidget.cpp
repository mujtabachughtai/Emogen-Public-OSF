#include "box_multiwidget.hpp"
#include <iostream>
#include <string>
box_multiwidget::box_multiwidget()
: m_button_check1("FACE 1"),
  m_button_check2("FACE 2"),
  m_button_check3("FACE 3"),
  m_button_check4("FACE 4"),
  m_button_check5("FACE 5"),
  m_button_check6("FACE 6"),
  m_button_check7("FACE 7"),
  m_button_check8("FACE 8"),
  m_button_check9("FACE 9"),
  m_button_check10("FACE 10"),
  m_button("NEXT GENERATION"),
  //m_button2("SAVE NOW"),
  m_button3("CLOSE TOOL"),
  m_button4("RESET SESSION"),
  m_Frame_Normal1("EMOTION GENERATION"),
  m_Frame_Normal("UNREALISTIC FACE REPORTING"),
  m_button_check1_ml("FACE 1"),
  m_button_check2_ml("FACE 2"),
  m_button_check3_ml("FACE 3"),
  m_button_check4_ml("FACE 4"),
  m_button_check5_ml("FACE 5"),
  m_button_check6_ml("FACE 6"),
  m_button_check7_ml("FACE 7"),
  m_button_check8_ml("FACE 8"),
  m_button_check9_ml("FACE 9"),
  m_button_check10_ml("FACE 10"),
  hello(),
  selections()
{

  hello.set_markup(TARGET_EMOTION);
  selections.set_markup(SELECTION_INSTRUCTIONS);
  // This just sets the title of our new window.
  set_title("Make your selections");

  
  // sets the border width of the window.
  set_border_width(20);

  m_button_check1.signal_clicked().connect(sigc::bind<Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 1"));
  m_box1.pack_start(m_button_check1);
  m_button_check1.show();

  m_button_check6.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 6"));
  m_box1.pack_start(m_button_check6);  
  m_button_check6.show();

  m_button_check2.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 2"));
  m_box2.pack_start(m_button_check2);  
  m_button_check2.show();

  m_button_check7.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 7"));
  m_box2.pack_start(m_button_check7);  
  m_button_check7.show();

  m_button_check3.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 3"));
  m_box3.pack_start(m_button_check3);  
  m_button_check3.show();

  m_button_check8.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 8"));
  m_box3.pack_start(m_button_check8);  
  m_button_check8.show();

  m_button_check4.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 4"));
  m_box4.pack_start(m_button_check4);  
  m_button_check4.show();

  m_button_check9.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 9"));
  m_box4.pack_start(m_button_check9);  
  m_button_check9.show();

  m_button_check5.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 5"));
  m_box5.pack_start(m_button_check5);  
  m_button_check5.show();

  m_button_check10.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 10"));
  m_box5.pack_start(m_button_check10);  
  m_button_check10.show();


  m_Entry.set_max_length(100);
  m_Entry.set_text("Enter best face number here (required)");
  m_Entry.select_region(0, m_Entry.get_text_length());
   
 
  m_button.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "NEXT GENERATION"));

  // disable NEXT GENERATION by default
  m_button.set_sensitive(false);

  m_box7.pack_start(m_button);  
  m_button.show(); 


 // SAVE BUTTON
 // ++++++++++++++++++++++++++++++++++++++++++++++++++++
 // m_button2.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
 //              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "SAVE NOW"));
 // m_box8.pack_start(m_button2);  
 // m_button2.show(); 

 // m_Entry_save_name.set_max_length(100);
 // m_Entry_save_name.set_text("Enter filename for chosen face");
 // m_Entry_save_name.select_region(0, m_Entry_save_name.get_text_length());
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++

  m_button3.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "CLOSE"));
  m_box9.pack_start(m_button3);  
  m_button3.show();

  m_button4.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "RESET SESSION"));
  m_box10.pack_start(m_button4);  
  m_button4.show();

    
  m_box6.pack_start(m_box1, true, false);
  m_box6.pack_start(m_box2, true, false);
  m_box6.pack_start(m_box3, true, false);
  m_box6.pack_start(m_box4, true, false);
  m_box6.pack_start(m_box5, true, false);
  
  
  m_box11.pack_start(hello, false, false,3);
  m_box11.pack_start(selections,false,false, 3);
  m_box11.pack_start(m_box6, false, false);
  m_box11.pack_start(m_Entry, false, false);
  m_box11.pack_start(m_box7, false, false);

  // SAVE BUTTON
  // ++++++++++++++++++++++++++++++
  //m_box11.pack_start(m_box8, false, false);
  //m_box11.pack_start(m_Entry_save_name, false, false);
  // ++++++++++++++++++++++++++++++

  m_box11.pack_start(m_box10, false, false);
  m_box11.pack_start(m_box9, false, false);


  m_box11.set_border_width(10);
  m_Frame_Normal1.add(m_box11);


  m_button_check1_ml.signal_clicked().connect(sigc::bind<Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 1"));
  m_box1_ml.pack_start(m_button_check1_ml);
  m_button_check1_ml.show();

  m_button_check6_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 6"));
  m_box1_ml.pack_start(m_button_check6_ml);  
  m_button_check6_ml.show();

  m_button_check2_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 2"));
  m_box2_ml.pack_start(m_button_check2_ml);  
  m_button_check2_ml.show();

  m_button_check7_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 7"));
  m_box2_ml.pack_start(m_button_check7_ml);  
  m_button_check7_ml.show();

  m_button_check3_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 3"));
  m_box3_ml.pack_start(m_button_check3_ml);  
  m_button_check3_ml.show();

  m_button_check8_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 8"));
  m_box3_ml.pack_start(m_button_check8_ml);  
  m_button_check8_ml.show();

  m_button_check4_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 4"));
  m_box4_ml.pack_start(m_button_check4_ml);  
  m_button_check4_ml.show();

  m_button_check9_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 9"));
  m_box4_ml.pack_start(m_button_check9_ml);  
  m_button_check9_ml.show();

  m_button_check5_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 5"));
  m_box5_ml.pack_start(m_button_check5_ml);  
  m_button_check5_ml.show();

  m_button_check10_ml.signal_clicked().connect(sigc::bind<-1, Glib::ustring>(
              sigc::mem_fun(*this, &box_multiwidget::on_button_clicked), "FACE 10"));
  m_box5_ml.pack_start(m_button_check10_ml);  
  m_button_check10_ml.show();

  m_box6_ml.pack_start(m_box1_ml, true, false);
  m_box6_ml.pack_start(m_box2_ml, true, false);
  m_box6_ml.pack_start(m_box3_ml, true, false);
  m_box6_ml.pack_start(m_box4_ml, true, false);
  m_box6_ml.pack_start(m_box5_ml, true, false);

  m_box6_ml.set_border_width(10);
  m_Frame_Normal.add(m_box6_ml); 
 

  m_box_main.pack_start(m_Frame_Normal1,false, false);
  m_box_main.pack_start(m_Frame_Normal,false, 5);

  add(m_box_main);


  show_all_children();

  
}

box_multiwidget::~box_multiwidget()
{
}

// Our new improved signal handler.  The data passed to this method is
// printed to stdout.
void box_multiwidget::on_button_clicked(Glib::ustring data)
{

     active_faces[0] = m_button_check1.get_active();
     active_faces[1] = m_button_check2.get_active();
     active_faces[2] = m_button_check3.get_active();
     active_faces[3] = m_button_check4.get_active();
     active_faces[4] = m_button_check5.get_active();
     active_faces[5] = m_button_check6.get_active();
     active_faces[6] = m_button_check7.get_active();
     active_faces[7] = m_button_check8.get_active();
     active_faces[8] = m_button_check9.get_active();
     active_faces[9] = m_button_check10.get_active();

     monster_faces[0] = m_button_check1_ml.get_active();
     monster_faces[1] = m_button_check2_ml.get_active();
     monster_faces[2] = m_button_check3_ml.get_active();
     monster_faces[3] = m_button_check4_ml.get_active();
     monster_faces[4] = m_button_check5_ml.get_active();
     monster_faces[5] = m_button_check6_ml.get_active();
     monster_faces[6] = m_button_check7_ml.get_active();
     monster_faces[7] = m_button_check8_ml.get_active();
     monster_faces[8] = m_button_check9_ml.get_active();
     monster_faces[9] = m_button_check10_ml.get_active();


     for (int k=0; k < active_faces.size(); ++k) std::cout << active_faces[k] << " ";
     std::cout << std::endl;

     eliteFace = atoi(m_Entry.get_text().c_str());
     
     std::cout << data << " was pressed" << std::endl;

     
     int numberOfnonSelections = std::count(active_faces.begin(), active_faces.end(), 0);
     int numberOfSelections = active_faces.size() - numberOfnonSelections; 

     //m_Entry.set_text("Enter best face number here (required).");
     if( (numberOfSelections >= min_num_of_sel) && (numberOfSelections <= max_num_of_sel) ) { 
           m_button.set_sensitive(true);  
     }else{ m_button.set_sensitive(false); }


     std::string signal = data.c_str();
     if (signal.compare("NEXT GENERATION") == 0 ) {
         std::cout << "Elite face: " << eliteFace << std::endl;
         if (eliteFace <= 0 || eliteFace > 10) {

           m_Entry.set_text("ERROR: No (valid) BEST FACE number entered.");

         } else if ( active_faces[eliteFace - 1] != 1) {
           
           m_Entry.set_text("ERROR: Entered number is not among the selections.");

         } else {

             exit_code = 0;
	     this -> close();
         }

     }

     if (signal.compare("SAVE NOW") == 0) {

           eliteFace_save_name = m_Entry_save_name.get_text().c_str(); 
           std::cout << eliteFace_save_name << " " << eliteFace_save_name.compare("Enter filename for chosen face") <<  std::endl;
           if (eliteFace <= 0 || eliteFace > 10)  {

              m_Entry.set_text("ERROR: No (valid) BEST FACE number to save.");

           } else if (eliteFace_save_name.empty() || !eliteFace_save_name.compare("Enter filename for chosen face") 
                                                  || !eliteFace_save_name.compare("ERROR: Please enter filename") ) {

              m_Entry_save_name.set_text("ERROR: Please enter filename");

           } else {
                 std::cout << "Best face will be saved under name.. " << eliteFace_save_name << std::endl;    
		 exit_code = 1;
                 this -> close();
           }
    
     }

     if (signal.compare("RESET SESSION") == 0) {

          exit_code = 2;
          this -> close();
     }

     if (signal.compare("CLOSE") == 0) {

          exit_code = 3;
          this -> close();
     }

}

