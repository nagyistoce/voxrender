/* ===========================================================================

	Project: VoxRender

	Description: Implements a control interface for 4D volume time steps

    Copyright (C) 2014 Lucas Sherman

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
#ifndef TIMING_WIDGET_H
#define TIMING_WIDGET_H

// QT Includes
#include <QtWidgets/QWidget>
#include "VoxScene/Scene.h"

namespace Ui { class TimingWidget; }

// Volume data histogram widget
class TimingWidget : public QWidget
{
	Q_OBJECT

public:
	explicit TimingWidget(QWidget *parent = 0);

	~TimingWidget();

private:
	Ui::TimingWidget *ui;

    void update();

    bool m_ignore;

private slots:
    void sceneChanged(vox::Scene & scene, void * userInfo);

    void on_doubleSpinBox_x_valueChanged(double value);
    void on_doubleSpinBox_y_valueChanged(double value);
    void on_doubleSpinBox_z_valueChanged(double value);
    void on_horizontalSlider_x_valueChanged(int value);
    void on_horizontalSlider_y_valueChanged(int value);
    void on_horizontalSlider_z_valueChanged(int value);
    void on_horizontalSlider_t_valueChanged(int value);
};

// End definition
#endif // SAMPLING_WIDGET_H

