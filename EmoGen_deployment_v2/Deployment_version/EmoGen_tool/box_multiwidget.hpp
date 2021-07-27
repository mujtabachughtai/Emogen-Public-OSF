#ifndef BOX_MULTIWIDGET_HPP
#define BOX_MULTIWIDGET_HPP

#include <gtkmm/box.h>
#include <gtkmm/button.h>
#include <gtkmm/window.h>
#include <gtkmm/checkbutton.h>
#include <gtkmm.h>
extern std::vector<bool> active_faces;
extern std::vector<bool> monster_faces;
extern int eliteFace;
extern std::string eliteFace_save_name;
extern int exit_code;
extern int min_num_of_sel, max_num_of_sel;
extern std::string TARGET_EMOTION;
extern std::string SELECTION_INSTRUCTIONS;
class box_multiwidget: public Gtk::Window
{
public:
  box_multiwidget();
  virtual ~box_multiwidget();

protected:

  // Signal handlers:
  // Our new improved on_button_clicked(). (see below)
  void on_button_clicked(Glib::ustring data);

  
  // Child widgets:
  Gtk::VBox m_box1;
  Gtk::VBox m_box2;
  Gtk::VBox m_box3;
  Gtk::VBox m_box4;
  Gtk::VBox m_box5;
  Gtk::HBox m_box6 = Gtk::HBox(false,5);
  Gtk::VBox m_box7;
  Gtk::VBox m_box8;
  Gtk::VBox m_box9;
  Gtk::VBox m_box10;
  Gtk::VBox m_box11;
  Gtk::VBox m_box12;
  Gtk::VBox m_box_main;
  Gtk::Button m_button;
  Gtk::Button m_button2;
  Gtk::Button m_button3;
  Gtk::Button m_button4;
  Gtk::CheckButton m_button_check1, m_button_check2, m_button_check3, m_button_check4, m_button_check5;
  Gtk::CheckButton m_button_check6, m_button_check7, m_button_check8, m_button_check9, m_button_check10;
  Gtk::Entry m_Entry;
  Gtk::Entry m_Entry_save_name;

  Gtk::Frame m_Frame_Normal, m_Frame_Normal1;
  Gtk::CheckButton m_button_check1_ml, m_button_check2_ml, m_button_check3_ml, m_button_check4_ml, m_button_check5_ml;
  Gtk::CheckButton m_button_check6_ml, m_button_check7_ml, m_button_check8_ml, m_button_check9_ml, m_button_check10_ml;
  Gtk::VBox m_box1_ml;
  Gtk::VBox m_box2_ml;
  Gtk::VBox m_box3_ml;
  Gtk::VBox m_box4_ml;
  Gtk::VBox m_box5_ml;
  Gtk::HBox m_box6_ml = Gtk::HBox(false,5);

   Gtk::Label hello;
   Gtk::Label selections;
};

#endif // BOX_MULTIWIDGET_HPP
