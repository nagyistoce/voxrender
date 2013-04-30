/* ===========================================================================

	Project: TransferWidget - Transfer Function Widget

	Description:
	 Implements an interface for interactive transfer function modification.

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
#ifndef TRANSFERWIDGET_H
#define TRANSFERWIDGET_H

// Include Dependencies
#include "histogramview.h"

// VoxLib Includes
#include "VoxLib/Core/VoxRender.h"

// QT Includes
#include <QWidget>
#include <QtGui/QColorDialog>

namespace Ui { class TransferWidget; }

// Transfer modification widget
class TransferWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit TransferWidget( QWidget *parent = 0 );
    ~TransferWidget( );
    
    void synchronizeView();
    void processInteractions();

private:
    Ui::TransferWidget *ui;

	// Color Selection Dialogue
	QColorDialog colorPicker;

	// Transfer function view
	HistogramView* m_primaryView;
	HistogramView* m_secondaryView;
	int            m_dimensions;

	// Current nodes
    std::shared_ptr<vox::Transfer> m_transfer;
    std::shared_ptr<vox::Node>     m_currentNode;
	//vox::Transfer::Region* m_currentRegion;
    //vox::Node*             m_currentNode;

	// Dimensions switching
	void switchDimensions( int nDims );
	bool canSwitchDimensions( );

signals:
	void transferChanged( );

public slots:
    void setSelectedNode(std::shared_ptr<vox::Node> node);

private slots:
	// Node selection group box
	void on_pushButton_delete_clicked( );
	void on_pushButton_next_clicked( );
	void on_pushButton_prev_clicked( );
	void on_pushButton_first_clicked( );
	void on_pushButton_last_clicked( );

	// Local transfer group box
	void on_pushButton_emission_clicked( );
	void on_pushButton_specular_clicked( );
	void on_pushButton_diffuse_clicked( );

	// Dimension selection group box
	void on_radioButton_1_toggled( bool checked );
	void on_radioButton_2_toggled( bool checked );
	void on_radioButton_3_toggled( bool checked );

    // Slider/SpinBox connections
    void on_doubleSpinBox_opacity_valueChanged(double value);
    void on_doubleSpinBox_density_valueChanged(double value);
    void on_doubleSpinBox_gloss_valueChanged(double value);
    void on_horizontalSlider_opacity_valueChanged(int value);
    void on_horizontalSlider_density_valueChanged(int value);
    void on_horizontalSlider_gloss_valueChanged(int value);
};

#endif // TRANSFERWIDGET_H
