/* ===========================================================================

	Project: VoxRender - Environment Light Interface

	Description: Implements the interface for ambient lighting

    Copyright (C) 2012 Lucas Sherman

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
#ifndef AMBIENT_LIGHT_WIDGET_H
#define AMBIENT_LIGHT_WIDGET_H

// QT4 Dependencies
#include <QtWidgets/QWidget>
#include <QtWidgets/QColorDialog>

// Generated class
namespace Ui { class AmbientLightWidget; }

// Scene Light class
namespace vox { class Light; }

// Point light interface
class AmbientLightWidget : public QWidget
{
	Q_OBJECT

public:
	explicit AmbientLightWidget(QWidget * parent);

	~AmbientLightWidget();

    QString title() { return m_title; }
    int index()     { return m_index; }

    void setIndex(int index) { m_index = index; }

    void processInteractions();
    void synchronizeView();

private:
	Ui::AmbientLightWidget* ui;

	QString m_title; ///< Light identifier/name

	int m_index;
    bool m_dirty;

private slots:
	void on_horizontalSlider_intensity_valueChanged(int value);
	void on_doubleSpinBox_intensity_valueChanged(double value);
};

#endif // AMBIENT_LIGHT_WIDGET_H

