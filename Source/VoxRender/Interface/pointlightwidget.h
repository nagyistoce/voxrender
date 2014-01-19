/* ===========================================================================

	Project: VoxRender - Point Light Interface

	Based on luxrender light group widget class.
	Lux Renderer website : http://www.luxrender.net 

	Description:
	 Implements the interface for point light source settings

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
#ifndef POINT_LIGHT_WIDGET_H
#define POINT_LIGHT_WIDGET_H

// QT Dependencies
#include <QtWidgets/QWidget>
#include <QtWidgets/QColorDialog>

#include "Extensions/QColorPushButton.h"

// Standard Library
#include <memory>

// Generated class
namespace Ui { class PointLightWidget; }

// Scene Light class
namespace vox { class Light; }

// Point light interface
class PointLightWidget : public QWidget
{
	Q_OBJECT

public:
	explicit PointLightWidget(QWidget * parent, std::shared_ptr<vox::Light> light);

	~PointLightWidget();

    QString title() { return m_title; }
    int index()     { return m_index; }

    void setIndex(int index) { m_index = index; }

    void processInteractions();

protected:
    virtual void changeEvent(QEvent * event);

private:
	Ui::PointLightWidget* ui;

	QString m_title; ///< Light identifier/name

    std::shared_ptr<vox::Light> m_light; ///< Associated scene object

	int m_index;
    bool m_dirty;

    QColorPushButton * m_colorButton;

private slots:
	void on_horizontalSlider_intensity_valueChanged(int value);
	void on_doubleSpinBox_intensity_valueChanged(double value);
	void on_horizontalSlider_latitude_valueChanged(int value);
	void on_doubleSpinBox_latitude_valueChanged(double value);
	void on_horizontalSlider_longitude_valueChanged(int value);
	void on_doubleSpinBox_longitude_valueChanged(double value);
	void on_horizontalSlider_distance_valueChanged(int value);
	void on_doubleSpinBox_distance_valueChanged(double value);
    
    void colorChanged(QColor const& color);
};

#endif // POINT_LIGHT_WIDGET_H

