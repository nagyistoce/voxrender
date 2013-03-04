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

// Include Headers
#include "transferwidget.h"
#include "ui_transferwidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// QT4 Includes
#include <QtGui/QMessageBox>

// File scope namespace
namespace 
{
    namespace filescope
    {
        // :TODO: Use a color selector item
        char const* stylesheet = "background-color: %1%";
    }
}

// ---------------------------------------------------------
// Constructor - Initialize widget slots and signals
// ---------------------------------------------------------
TransferWidget::TransferWidget( QWidget *parent ) :
    QWidget( parent ),
    ui( new Ui::TransferWidget ),
    m_currentRegion(nullptr),
    m_currentNode(nullptr)
{
    ui->setupUi(this);

	// Transfer function views
	m_primaryView = new HistogramView( ui->transferPrimary );
	ui->gridLayout_transferPrimary->addWidget( m_primaryView, 0, 0, 1, 1 );
	m_secondaryView = new HistogramView( ui->transferSecondary );
	ui->gridLayout_transferSecondary->addWidget( m_secondaryView, 0, 0, 1, 1 );

	// Default transfer function
	switchDimensions( 1 );
}
    
// ---------------------------------------------------------
// Destructor - Delete transfer views
// ---------------------------------------------------------
TransferWidget::~TransferWidget( )
{
    delete ui;
}

// ---------------------------------------------------------
// Returns true if the user allows dimension change
// ---------------------------------------------------------
bool TransferWidget::canSwitchDimensions( )
{
	QMessageBox msgBox(this);
	msgBox.setIcon( QMessageBox::Question );
	msgBox.setText( tr("Are you sure you want to switch dimension?\
					Any unsaved changes to the current function will be lost.") );
	msgBox.setWindowTitle( "Switch Dimensions" );

	QPushButton* accept = msgBox.addButton( tr("Yes"), QMessageBox::AcceptRole );
	msgBox.addButton( tr("No"), QMessageBox::RejectRole );
	QPushButton* cancel = msgBox.addButton( tr("Cancel"), QMessageBox::RejectRole );
	msgBox.setDefaultButton( cancel );

	msgBox.exec( );

	if( msgBox.clickedButton( ) != accept ) return false;

	return true;
}

// ---------------------------------------------------------
// Blocks radio button signals 
// ---------------------------------------------------------
void TransferWidget::switchDimensions( int nDims )
{
	// Block radio button signals
	ui->radioButton_1->blockSignals( true );
	ui->radioButton_2->blockSignals( true );
	ui->radioButton_3->blockSignals( true );

	// Switch transfer control context
	switch( nDims ) 
	{
		case 1:

			// Configure widget for 1-Dimensional work 
			ui->radioButton_1->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->hide( );
			ui->doubleSpinBox_gradient->hide( );
			ui->label_gradient->hide( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 1;
			break;

		case 2:

			// Configure widget for 2-Dimensional work
			ui->radioButton_2->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 2;
			break;

		case 3:

			// Configure widget for 3-Dimensional work
			ui->radioButton_3->setChecked( true ); 
			ui->groupBox_transferSecondary->show( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->show( );
			ui->doubleSpinBox_gradient2->show( );
			ui->label_gradient2->show( );
			m_dimensions = 3;
			break; 

        default:
            break;
	}

	// Unblock radio button signals
	ui->radioButton_1->blockSignals( false );
	ui->radioButton_2->blockSignals( false );
	ui->radioButton_3->blockSignals( false );
}

// ---------------------------------------------------------
// Single dimension transfer selection
// ---------------------------------------------------------
void TransferWidget::on_radioButton_1_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(1);
}

// ---------------------------------------------------------
// Single dimension transfer selection
// ---------------------------------------------------------
void TransferWidget::on_radioButton_2_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(2);
}

// ---------------------------------------------------------
// Single dimension transfer selection
// ---------------------------------------------------------
void TransferWidget::on_radioButton_3_toggled(bool checked)
{	
	if (checked && canSwitchDimensions()) switchDimensions(3);
}

// ---------------------------------------------------------
// Select first node 
// ---------------------------------------------------------
void TransferWidget::on_pushButton_first_clicked( )
{
    auto transfer = MainWindow::instance->activeScene.transfer;
    m_currentRegion = transfer->regions( ).front( );
    m_currentNode = &m_currentRegion->nodes[0];
}

// ---------------------------------------------------------
// Select next node 
// ---------------------------------------------------------
void TransferWidget::on_pushButton_next_clicked( )
{
}

// ---------------------------------------------------------
// Select prev node 
// ---------------------------------------------------------
void TransferWidget::on_pushButton_prev_clicked( )
{
}

// ---------------------------------------------------------
// Select last node 
// ---------------------------------------------------------
void TransferWidget::on_pushButton_last_clicked( )
{
    auto transfer = MainWindow::instance->activeScene.transfer;
    m_currentRegion = transfer->regions( ).back( );
    m_currentNode = &m_currentRegion->nodes[0];
}

// ---------------------------------------------------------
// Delete selected node
// ---------------------------------------------------------
void TransferWidget::on_pushButton_delete_clicked( )
{
    auto transfer = MainWindow::instance->activeScene.transfer;
    transfer->removeRegion( m_currentRegion );

    on_pushButton_first_clicked( );
}

// ---------------------------------------------------------
// Select local emission transfer
// ---------------------------------------------------------
void TransferWidget::on_pushButton_emission_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_emission->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// ---------------------------------------------------------
// Select local specular transfer
// ---------------------------------------------------------
void TransferWidget::on_pushButton_specular_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_specular->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// ---------------------------------------------------------
// Select local diffuse transfer
// ---------------------------------------------------------
void TransferWidget::on_pushButton_diffuse_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_diffuse->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_density_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_density,
        ui->horizontalSlider_density,
        value);
}

// --------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// --------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_density_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_density,
        ui->doubleSpinBox_density,
        value);
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_gloss_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_gloss,
        ui->horizontalSlider_gloss,
        value);
}

// --------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// --------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_gloss_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_gloss,
        ui->doubleSpinBox_gloss,
        value);
}

// --------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// --------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_opacity_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_opacity,
        ui->horizontalSlider_opacity,
        value);
}

// --------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// --------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_opacity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_opacity,
        ui->doubleSpinBox_opacity,
        value);
}

