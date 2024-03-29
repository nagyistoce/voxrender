/* ===========================================================================

	Project: SamplingWidget - Sampling widget

	Description:
	 Implements an interface for modifying the sampling parameters of the
	 volume ray tracer.

    Copyright (C) 2012-2014 Lucas Sherman

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
#ifndef SAMPLING_WIDGET_H
#define SAMPLING_WIDGET_H

// QT Includes
#include <QtWidgets/QWidget>
#include "VoxScene/Scene.h"

namespace Ui { class SamplingWidget; }

// Volume data histogram widget
class SamplingWidget : public QWidget
{
	Q_OBJECT

public:
	explicit SamplingWidget(QWidget *parent = 0);

	~SamplingWidget();

private:
	Ui::SamplingWidget *ui;

    void update();

    bool m_ignore;

signals:
	void valuesChanged();

private slots:
	void on_horizontalSlider_gradient_valueChanged(int value);
	void on_doubleSpinBox_gradient_valueChanged(double value);
	void on_horizontalSlider_primaryStep_valueChanged(int value);
	void on_doubleSpinBox_primaryStep_valueChanged(double value);
	void on_horizontalSlider_shadowStep_valueChanged(int value);
	void on_doubleSpinBox_shadowStep_valueChanged(double value);
	void on_doubleSpinBox_coefficient_valueChanged(double value);
	void on_horizontalSlider_coefficient_valueChanged(int value);
	void on_doubleSpinBox_edge_valueChanged(double value);
    void on_horizontalSlider_edge_valueChanged(int value);

    void sceneChanged(vox::Scene & scene, void * userInfo);
};

// End definition
#endif // SAMPLING_WIDGET_H

