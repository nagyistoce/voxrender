/* ===========================================================================

	Project: VoxRender

	Description: Implements the interface for manually editing 

    Copyright (C) 2013 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin Definition
#ifndef CLIP_PLANE_WIDGET_H
#define CLIP_PLANE_WIDGET_H

// QT Dependencies
#include <QtWidgets/QWidget>

// Standard Library
#include <memory>

// Generated class
namespace Ui { class ClipPlaneWidget; }

// Scene clip plane class
namespace vox { class Plane; }

/** Widget for controlling clipping planes in the scene */
class ClipPlaneWidget : public QWidget
{
	Q_OBJECT

public:
    /** Constructs a new widget with an associated plane object in the scene */
	explicit ClipPlaneWidget(QWidget * parent, void * userInfo, std::shared_ptr<vox::Plane> plane);

    /** Destructor */
	~ClipPlaneWidget();

    /** Returns the clip plane associated with this control */
    std::shared_ptr<vox::Plane> plane() { return m_plane; }

    /** Called when the plane is modified externally to synchronize the controls */
    void sceneChanged();

private:
    /** Synchronizes the clip plane with the widget's controls */
    void update();
    
private:
	Ui::ClipPlaneWidget* ui;

    std::shared_ptr<vox::Plane> m_plane; ///< Associated scene object

    bool m_ignore; ///< Simple blockSignals alternative
    
    void * m_userInfo; ///< User info for scene change events

protected:
    void changeEvent(QEvent * event);

private slots:
    void on_horizontalSlider_pitch_valueChanged(int value);
    void on_horizontalSlider_yaw_valueChanged(int value);
    void on_horizontalSlider_x_valueChanged(int value);
    void on_horizontalSlider_y_valueChanged(int value);
    void on_horizontalSlider_z_valueChanged(int value);

    void on_doubleSpinBox_pitch_valueChanged(double value);
    void on_doubleSpinBox_yaw_valueChanged(double value);
    void on_doubleSpinBox_x_valueChanged(double value);
    void on_doubleSpinBox_y_valueChanged(double value);
    void on_doubleSpinBox_z_valueChanged(double value);
};

#endif // CLIP_PLANE_WIDGET_H

