package ch08_interface.sec13_sealed_interface;

public sealed interface InterfaceA permits InterfaceB {
	void methodA();
}